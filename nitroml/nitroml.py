# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Lint as: python3
"""NitroML: Accelerate AutoML research.

go/nitroml

NitroML is a framework for benchmarking AutoML workflows composed of
TFX OSS components.

NitroML enables AutoML teams to iterate more quickly on their custom machine
learning pipelines. It offers machine learning benchmarking best practices
out-of-the-box, curates public datasets, and scales with Google-resources.
Its benchmark database and analysis tools ensure that AutoML teams can be
data-driven as they modify their systems.

NitroML's API is inspired by the Python stdlib's `unittest` package. It
is intended to encourage writing benchmarks in a similar manner to writing
test cases.
"""

import abc
import contextlib
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from absl import app
from absl import flags
from absl import logging
from nitroml import pipeline_filtering
from nitroml.benchmark.task import BenchmarkTask
from nitroml.subpipeline import Subpipeline
from nitroml.subpipeline import SubpipelineOutputs
import tensorflow as tf
from tfx.dsl.compiler import compiler
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration.portable import tfx_runner as tfx_runner_lib

from ml_metadata.proto import metadata_store_pb2

T = TypeVar("T")

# List, Tuple, and Dict must contain PipelineLike values, but recursive type
# annotations are not yet supported in Pytype.
PipelineLike = Union[BaseComponent, Subpipeline, List, Tuple, Dict]

FLAGS = flags.FLAGS

# FLAGS

# TODO(b/168906137): Eliminate `match` flag, once we have programmatic partial
# run support for partially executing a TFX DAG.
flags.DEFINE_string(
    "match", "",
    "Specifies a regex to match and filter benchmarks. For example, passing "
    '`--match=".*mnist.*"` will execute only the benchmarks whose names '
    'contain the substring "mnist" and skip the rest.')
flags.DEFINE_integer(
    "runs_per_benchmark", None,
    "Specifies the number of times each benchmark should be executed. The "
    "benchmarks' pipelines are concatenated into a single DAG so that the "
    "orchestrator can run them in parallel. For example, passing "
    "`--runs_per_benchmark=5` will execute each benchmark 5 times in "
    "parallel. When calling nitroml.results.overview(), metrics in benchmark "
    "run results can be optionally aggregated to compute means, standard "
    "deviations, and other aggregate metrics.")


def _validate_regex(regex: str) -> bool:
  try:
    re.compile(regex)
    return True
  except re.error:
    return False


flags.register_validator(
    "match", _validate_regex, message="--match must be a valid regex.")


def _runs_suffix(benchmark_run: int, runs_per_benchmark: int) -> str:
  runs = runs_per_benchmark
  return f".run_{benchmark_run}_of_{runs}" if runs > 1 else ""


def _qualified_name(prefix: str, name: str, benchmark_run: int,
                    runs_per_benchmark: int) -> str:
  name = "{}.{}".format(prefix, name) if prefix else name
  return name + _runs_suffix(benchmark_run, runs_per_benchmark)


class BenchmarkSubpipeline(Subpipeline):
  """A model-quality benchmark Subpipeline."""

  def __init__(self, benchmark_name: str, components: List[BaseComponent]):
    self._benchmark_name = benchmark_name
    self._components = components

  @property
  def id(self) -> str:
    return self._benchmark_name

  @property
  def components(self) -> List[BaseComponent]:
    return self._components

  @property
  def outputs(self) -> SubpipelineOutputs:
    # Returns no outputs.
    return SubpipelineOutputs({})


class BenchmarkPipeline(Pipeline):
  """A TFX Pipeline composed of multiple benchmark subpipelines."""

  def __init__(self,
               components_to_always_add: List[BaseComponent],
               benchmark_subpipelines: List[BenchmarkSubpipeline],
               pipeline_name: Optional[str],
               pipeline_root: Optional[str],
               metadata_connection_config: Optional[
                   metadata_store_pb2.ConnectionConfig] = None,
               beam_pipeline_args: Optional[List[str]] = None,
               **kwargs):

    if not benchmark_subpipelines and not components_to_always_add:
      raise ValueError(
          "Requires at least one benchmark subpipeline or component to run. "
          "You may want to call `self.add(..., always=True) in order "
          "to run Components, Subpipelines, or Pipeline even without requiring "
          "a call to `self.evaluate(...)`.")

    # Set defaults.
    if not pipeline_name:
      pipeline_name = "nitroml"
    if not pipeline_root:
      tmp_root_dir = os.path.join("/tmp", pipeline_name)
      tf.io.gfile.makedirs(tmp_root_dir)
      pipeline_root = tempfile.mkdtemp(dir=tmp_root_dir)
      logging.info("Creating tmp pipeline_root at %s", pipeline_root)
    if not metadata_connection_config:
      metadata_connection_config = metadata_store_pb2.ConnectionConfig(
          sqlite=metadata_store_pb2.SqliteMetadataSourceConfig(
              filename_uri=os.path.join(pipeline_root, "mlmd.sqlite")))

    # Ensure that pipeline dirs are created.
    _make_pipeline_dirs(pipeline_root, metadata_connection_config)

    components = set(components_to_always_add)
    for benchmark_subpipeline in benchmark_subpipelines:
      for component in benchmark_subpipeline.components:
        components.add(component)
    super().__init__(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=metadata_connection_config,
        components=list(components),
        beam_pipeline_args=beam_pipeline_args,
        **kwargs)

    self._subpipelines = benchmark_subpipelines

  @property
  def benchmark_names(self) -> List[str]:
    return [p.id for p in self._subpipelines]


def _make_pipeline_dirs(
    pipeline_root: str,
    metadata_connection_config: metadata_store_pb2.ConnectionConfig) -> None:
  """Makes the relevent dirs if needed."""

  tf.io.gfile.makedirs(pipeline_root)
  if metadata_connection_config.HasField("sqlite"):
    tf.io.gfile.makedirs(
        os.path.dirname(metadata_connection_config.sqlite.filename_uri))


class BenchmarkSpec:
  """Holder for benchmark call information.

  BenchmarkSpecs are automatically managed by the Benchmark class, and do
  not need to be explicitly manipulated by benchmark authors.
  """

  def __init__(self, benchmark_run: int, runs_per_benchmark: int):
    self.components_to_always_add = []
    self.requested_partial_run = False
    self.components_to_partial_run = []
    self.exclusive_components_to_partial_run = []
    self.benchmark_subpipelines = []
    self.benchmark_run = benchmark_run
    self.runs_per_benchmark = runs_per_benchmark

  @property
  def components(self) -> List[BaseComponent]:
    """Returns the components produced when calling a Benchmark."""

    components = self.components_to_always_add
    for subpipeline in self.benchmark_subpipelines:
      components.extend(subpipeline.components)
    return list(set(components))

  @property
  def nodes_to_partial_run(self) -> List[str]:
    nodes_to_run = self.exclusive_components_to_partial_run or self.components_to_partial_run
    return [c.id for c in nodes_to_run]


class Benchmark(abc.ABC):
  """A benchmark which can be composed of several benchmark methods.

  The Benchmark object design is inspired by `unittest.TestCase`.

  A benchmark file can contain multiple Benchmark subclasses to compose a suite
  of benchmarks.
  """

  def __init__(self):
    self._benchmark = self  # The sub-benchmark stack.
    self._spec = None
    self._components = []

  @abc.abstractmethod
  def benchmark(self, **kwargs):
    """Benchmark method to be overridden by subclasses.

    Args:
      **kwargs: Keyword args that are propagated from the called to
        nitroml.run(...).
    """

  @contextlib.contextmanager
  def sub_benchmark(self, name: str):
    """Executes the enclosed code block as a sub-benchmark.

    A benchmark can contain any number of sub-benchmark declarations, and they
    can be arbitrarily nested.

    Args:
      name: String which is appended to the benchmark's name which can be used
        for filtering (b/143771302). name is also displayed whenever a
        sub-benchmark raises an exception, allowing the user to identify it.

    Yields:
      A context manager which executes the enclosed code block as a
      sub-benchmark.
    """

    benchmark = self._benchmark
    self._benchmark = _SubBenchmark(benchmark, name)
    try:
      yield
    finally:
      self._benchmark = benchmark

  def _all_components(self) -> List[BaseComponent]:
    """Returns the components currently registered to this Benchmark."""

    return self._components

  def evaluate(self,
               task: BenchmarkTask,
               always: bool = False,
               only: bool = False,
               skip: bool = False,
               **kwargs) -> None:
    """Adds a benchmark subgraph to the benchmark suite's workflow DAG.

    Appends BenchmarkTask evaluation components to the given model in order to
    evaluate it.

    Requires the user to call `self.add(...)` on any components on the path to
    this evaluation.

    If the current pipeline does not produce a model, the caller should call
    `self.add(...)` with the `always=True` for all components in the pipeline.

    Args:
      task: A `BenchmarkTask` subclass instance that specifies the evaluations.
      always: See `BenchmarkTask#add(..., always=...)`.
      only: See `BenchmarkTask#add(..., only=...)`.
      skip: See `BenchmarkTask#add(..., skip=...)`.
      **kwargs: Additional kwargs to pass to `Task#make_evaluation`.
    """

    # Strip common parts of benchmark names from benchmark ID.
    # TODO(b/168906137): Much of this complexity can be removed once we have
    # partial run support in the IR.
    benchmark_name = self._benchmark.id() + _runs_suffix(
        self._spec.benchmark_run, self._spec.runs_per_benchmark)
    seen_benchmarks = set(p.id for p in self._spec.benchmark_subpipelines)
    if benchmark_name in seen_benchmarks:
      raise ValueError("evaluate was already called once for this benchmark. "
                       "Consider calling `with self.sub_benchmark(...):` and "
                       "then calling `self.evaluate(...)` within its scope.")

    self.add(
        task.make_evaluation(
            benchmark_name=self._benchmark.id(),
            benchmark_run=self._spec.benchmark_run,
            runs_per_benchmark=self._spec.runs_per_benchmark,
            **kwargs),
        always=always,
        only=only,
        skip=skip)

    self._spec.benchmark_subpipelines.append(
        BenchmarkSubpipeline(
            benchmark_name, components=self._benchmark._all_components()))  # pylint: disable=protected-access

  def id(self) -> str:
    """The unique ID of this benchmark."""

    return f"{self.__class__.__qualname__}.benchmark"

  def add(self,
          pipeline: PipelineLike,
          always: bool = False,
          only: bool = False,
          skip: bool = False) -> Any:
    """Adds the given nodes to the current benchmark DAG scope.

    There are several ways `self.add(...)` can be called. Suppose you have two
    `tfx.BaseComponent` subclasses, `MyExampleGen` and `MyTrainer`. The
    following are equivalent:

        example_gen, trainer = self.add((MyExampleGen(), MyTrainer()))

    and

        example_gen = self.add(MyExampleGen())
        trainer = self.add(MyTrainer())

    and

        example_gen = MyExampleGen())
        trainer = MyTrainer()
        self.add((example_gen, trainer))

    This method can also be called with a `nitroml.Subpipeline`. For example:

        my_subpipeline = self.add(MySubpipeline())

    When called within a `with self.sub_benchmark(...):` block, the current
    scope of the subbenchmark is appended to the Component instance names, so
    that the user does not need to manually set the instance name of each
    component.

    In order to run a DAG that does not produce a model, in which case you
    cannot call self.evaluate(), instead call self.add(..., always=True).

    This method also enables programmatic partial runs via the `skip` and `only`
    args. For example, in the following pipeline:

        example_gen = self.add(MyExampleGen())
        trainer = self.add(MyTrainer())

    Assuming you had a previous execution of example_gen in your MLMD instance,
    you can skip the example_gen stage execution, and reuse previously computed
    artifacts using `skip=True`:

        example_gen = self.add(MyExampleGen(), skip=True)
        trainer = self.add(MyTrainer())

    Alternatively, if you only want to execute select components, use
    `only=True`:

        example_gen = self.add(MyExampleGen())
        trainer = self.add(MyTrainer(), only=True)

    Args:
      pipeline: A tfx.BaseComponent, nitroml.Subpipeline, or tfx.Pipeline
        subclass instance, or nested dict, list, or tuple of these objects. The
        components that compose them will be registered to the benchmark DAG to
        be executed by the TFX Runner.
      always: When `True`, signals to always run the given components in
        `pipeline`, even when a subbenchmark is filtered with the `match`
        flag. TODO(b/168906137): Remove once we have built-in partial DAG run
        support.
      only: When `True`, only components and subpipelines added with `only=True`
        will be run, whereas other components will be skipped.
      skip: When `True`, only components and subpipelines added without
        `skip=True` will be run.

    Returns:
      The argument passed to `pipeline`.

    Raises:
      ValueError: When both `skip` and `only` are `True`.
    """

    if skip and only:
      raise ValueError("Only one of `skip` or `only` can be True.")

    # Recursive PipelineLike types.
    if isinstance(pipeline, list):
      return [
          self.add(x, always=always, only=only, skip=skip) for x in pipeline
      ]

    if isinstance(pipeline, tuple):
      return tuple(
          self.add(x, always=always, only=only, skip=skip) for x in pipeline)

    if isinstance(pipeline, dict):
      return {
          k: self.add(v, always=always, only=only, skip=skip)
          for k, v in pipeline.items()
      }

    components = []
    if isinstance(pipeline, BaseComponent):
      components.append(pipeline)
    elif isinstance(pipeline, Subpipeline):
      components += pipeline.components
    else:
      raise ValueError(f"Unsupported type for `pipeline`: {pipeline}")

    # All this subbenchmark's parents' and own components.
    # TODO(b/168906137): Much of this complexity can be removed once we have
    # partial run support in the IR.
    preexisting_components = frozenset(self._benchmark._all_components())  # pylint: disable=protected-access
    new_components = set()
    for component in components:
      if component in new_components or component in preexisting_components:
        continue
      # pylint: disable=protected-access
      component._instance_name = _qualified_name(
          prefix=component._instance_name,
          name=self._benchmark.id(),
          benchmark_run=self._spec.benchmark_run,
          runs_per_benchmark=self._spec.runs_per_benchmark)
      # pylint: enable=protected-access
      new_components.add(component)

    self._benchmark._components += list(new_components)
    if always:
      self._spec.components_to_always_add += list(new_components)
    if skip or only:
      self._spec.requested_partial_run = True
    if not skip:
      self._spec.components_to_partial_run += list(new_components)
    if only:
      self._spec.exclusive_components_to_partial_run += list(new_components)

    return pipeline

  def __call__(self, benchmark_run: int, runs_per_benchmark: int, *args,
               **kwargs):
    spec = BenchmarkSpec(benchmark_run, runs_per_benchmark)
    self._spec = spec
    try:
      self.benchmark(**kwargs)
    finally:
      self._spec = None
    return spec


class _SubBenchmark(Benchmark):
  """A benchmark nested within another benchmark.

  SubBenchmarks allow pipelines to split creating a tree-like workflow where
  each leaf node is evaluated independently. This allows artifacts to be shared
  between benchmarks for maximum resource efficiency.
  """

  def __init__(self, parent: Benchmark, name: str):
    super(_SubBenchmark, self).__init__()
    self._parent = parent
    self._name = name

  def _all_components(self) -> List[BaseComponent]:
    """Returns all this subbenchmark's parents' and own components."""

    return self._parent._all_components() + self._components  # pylint: disable=protected-access

  def benchmark(self):
    # Implement abstract method so that _SubBenchmarks can be instantiated
    # directly.
    raise RuntimeError("Sub-benchmarks are not intended to be called directly.")

  def id(self) -> str:
    return f"{self._parent.id()}.{self._name}"


def _get_subclasses(baseclass: T) -> List[T]:
  """Returns a list of subclasses of given baseclass.

  Args:
    baseclass: Any type of class. i.e _get_subclasses(AwesomeClass)

  Returns:
    A list contains all subclasses from all loader module including abstract
    implementation of baseclass.
  """
  subs = baseclass.__subclasses__()
  if not subs:
    return list()

  subsubs = list()
  for sub in subs:
    subsubs += _get_subclasses(sub)
  return subs + subsubs


def _load_benchmarks() -> List[Benchmark]:
  """Loads all subclasses of Benchmark from all loaded modules."""
  subclasses = _get_subclasses(Benchmark)
  if _SubBenchmark in subclasses:  # Internal subclass of Benchmark.
    subclasses.remove(_SubBenchmark)
  return [subclass() for subclass in subclasses]  # pylint: disable=no-value-for-parameter


def run(benchmarks: List[Benchmark],
        tfx_runner: Optional[tfx_runner_lib.TfxRunner] = None,
        pipeline_name: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
        enable_cache: Optional[bool] = False,
        beam_pipeline_args: Optional[List[str]] = None,
        **kwargs) -> BenchmarkPipeline:
  """Runs the given benchmarks as part of a single pipeline DAG.

  First it concatenates all the benchmark pipelines into a single DAG
  benchmark pipeline. Next it executes the workflow via tfx_runner.run().

  When the `match` flag is set, matched benchmarks are filtered by name.

  When the `runs_per_benchmark` flag is set, each benchmark is run the number
  of times specified.


  Args:
    benchmarks: List of Benchmark instances to include in the suite.
    tfx_runner: The TfxRunner instance that defines the platform where
      benchmarks are run.
    pipeline_name: Name of the benchmark pipeline.
    pipeline_root: Path to root directory of the pipeline.
    metadata_connection_config: The config to connect to ML metadata.
    enable_cache: Whether or not cache is enabled for this run.
    beam_pipeline_args: Beam pipeline args for beam jobs within executor.
      Executor will use beam DirectRunner as Default.
    **kwargs: Additional kwargs forwarded as kwargs to benchmarks.

  Returns:
    Returns the BenchmarkPipeline that was passed to the tfx_runner.

  Raises:
    ValueError: If the given tfx_runner is not supported.
  """

  if "compile_pipeline" in kwargs:
    kwargs.pop("compile_pipeline")
    logging.warning("The `compile_pipeline` argument DEPRECATED and ignored. "
                    "Pipelines are now automatically compiled.")

  runs_per_benchmark = FLAGS.runs_per_benchmark
  if runs_per_benchmark is None:
    runs_per_benchmark = int(os.environ.get("NITROML_RUNS_PER_BENCHMARK", 1))


  if not tfx_runner:
    logging.info("Setting TFX runner to OSS default: BeamDagRunner.")
    tfx_runner = beam_dag_runner.BeamDagRunner()

  if runs_per_benchmark <= 0:
    raise ValueError("runs_per_benchmark must be strictly positive; "
                     f"got runs_per_benchmark={runs_per_benchmark} instead.")

  benchmark_subpipelines = []
  for b in benchmarks:
    for benchmark_run in range(runs_per_benchmark):
      # Call benchmarks with pipeline args.
      spec = b(
          benchmark_run=benchmark_run + 1,
          runs_per_benchmark=runs_per_benchmark,
          **kwargs)
      for benchmark_subpipeline in spec.benchmark_subpipelines:
        if re.match(FLAGS.match, benchmark_subpipeline.id):
          benchmark_subpipelines.append(benchmark_subpipeline)

  if FLAGS.match and not benchmark_subpipelines:
    if spec.components_to_always_add:
      logging.info(
          "No benchmarks matched the pattern '%s'. "
          "Running components passed to self.add(..., always=True) only.",
          FLAGS.match)
    else:
      raise ValueError(f"No benchmarks matched the pattern '{FLAGS.match}'")

  benchmark_pipeline = BenchmarkPipeline(
      components_to_always_add=spec.components_to_always_add,
      benchmark_subpipelines=benchmark_subpipelines,
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=metadata_connection_config,
      enable_cache=enable_cache,
      beam_pipeline_args=beam_pipeline_args,
      **kwargs)

  logging.info("NitroML benchmarks:")
  for benchmark_name in benchmark_pipeline.benchmark_names:
    logging.info("\t%s", benchmark_name)
    logging.info("\t\tRUNNING")
  dsl_compiler = compiler.Compiler()
  pipeline_to_run = dsl_compiler.compile(benchmark_pipeline)
  if spec.requested_partial_run:
    logging.info("Only running the following nodes:\n%s",
                 "\n".join(spec.nodes_to_partial_run))
    pipeline_to_run = pipeline_filtering.filter_pipeline(
        input_pipeline=pipeline_to_run,
        pipeline_run_id_fn=(
            pipeline_filtering.make_latest_resolver_pipeline_run_id_fn(
                benchmark_pipeline.metadata_connection_config)),
        skip_nodes=lambda x: x not in set(spec.nodes_to_partial_run))

  tfx_runner.run(pipeline_to_run)
  return benchmark_pipeline


def main(*args, **kwargs) -> None:
  """Runs all available Benchmarks.

  Usually this function is called without arguments, so the
  nitroml.run() will get be called with the default settings, and it will
  will run all benchmark methods of all Benchmark classes in the __main__
  module.

  Executes all subclasses of Benchmark from all loaded modules.

  NOTE TO MAINTAINERS:
  There are clients which invoke this function directly; some explicitly set
  some of the **kwargs parameters, so care must be taken not to override them.

  Args:
    *args: Positional arguments passed to nitroml.run.
    **kwargs: Keyword arguments arguments passed to nitroml.run.
  """

  del args  # Unused

  def _main(argv):
    del argv  # Unused

    run(_load_benchmarks(), **kwargs)
    # Explicitly returning None.
    # Any other value than None or zero is considered “abnormal termination”.
    # https://docs.python.org/3/library/sys.html#sys.exit
    return None

  app.run(_main)
