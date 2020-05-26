# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""NitroML: Accelerate AutoML development.

go/nitroml

NitroML is a framework for benchmarking AutoML workflows composed of
TFX OSS and MyOrchestrator components.

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
import re
from typing import List, Text, TypeVar

from absl import app
from absl import flags
from absl import logging

from nitroml.components.publisher.component import BenchmarkResultPublisher
from my_orchestrator.common import my_orchestrator
from tfx import components as tfx
from tfx import types
from tfx.components.base import base_component

T = TypeVar("T")
pipeline_args = my_orchestrator.pipeline_args

FLAGS = flags.FLAGS

# FLAGS
flags.DEFINE_string(
    "match", "",
    "Specifies a regex to match and filter benchmarks. For example, passing "
    '`--match=".*mnist.*"` will execute only the benchmarks whose names '
    'contain the substring "mnist" and skip the rest.')
flags.DEFINE_integer(
    "runs_per_benchmark", 1,
    "Specifies the number of times each benchmark should be executed. The "
    "benchmarks' pipelines are concatenated into a single DAG so that the "
    "orchestrator can run them in parallel. For example, passing "
    "`--runs_per_benchmark=5` will execute each benchmark 5 times in "
    "parallel. When calling nitroml.results.overview(), metrics in benchmark "
    "run results can be optionally aggregated to compute means, standard "
    "deviations, and other aggregate metrics.")


def _validate_regex(regex: Text) -> bool:
  try:
    re.compile(regex)
    return True
  except re.error:
    return False


flags.register_validator(
    "match", _validate_regex, message="--match must be a valid regex.")


def _qualified_name(prefix: Text, name: Text) -> Text:
  return "{}.{}".format(prefix, name) if prefix else name


class _BenchmarkPipeline(object):
  """A pipeline for a benchmark."""

  def __init__(self, benchmark_name: Text,
               base_pipeline: List[base_component.BaseComponent],
               evaluator: tfx.Evaluator):
    self._benchmark_name = benchmark_name
    self._base_pipeline = base_pipeline
    self._evaluator = evaluator

  @property
  def benchmark_name(self) -> Text:
    return self._benchmark_name

  @property
  def base_pipeline(self) -> List[base_component.BaseComponent]:
    return self._base_pipeline

  @property
  def evaluator(self) -> tfx.Evaluator:
    return self._evaluator

  @property
  def pipeline(self) -> List[base_component.BaseComponent]:
    return self._base_pipeline + [self._evaluator]


class _RepeatablePipeline(object):
  """A repeatable benchmark."""

  def __init__(self, benchmark_pipeline: _BenchmarkPipeline, repetition: int,
               num_repetitions: int):
    self.benchmark_pipeline = benchmark_pipeline
    self._repetition = repetition
    self._num_repetitions = num_repetitions
    self._publisher = None

  @property
  def benchmark_name(self) -> Text:
    name = self.benchmark_pipeline.benchmark_name
    if self._num_repetitions == 1:
      return name
    return f"{name}.run_{self._repetition}_of_{self._num_repetitions}"

  @property
  def publisher(self) -> BenchmarkResultPublisher:
    if not self._publisher:
      self._publisher = BenchmarkResultPublisher(
          self.benchmark_name,
          self.benchmark_pipeline.evaluator.outputs.evaluation,
          run=self._repetition,
          num_runs=self._num_repetitions)
    return self._publisher

  @property
  def pipeline(self) -> List[base_component.BaseComponent]:
    return self.benchmark_pipeline.pipeline + [self.publisher]


class _ConcatenatedPipeline(object):
  """A pipeline composed of repeatable benchmark pipelines.

  For combining multiple benchmarked pipelines into a single DAG.
  """

  def __init__(self, pipelines: List[_RepeatablePipeline]):
    self._pipelines = pipelines

  @property
  def benchmark_names(self):
    return [p.benchmark_name for p in self._pipelines]

  def __call__(self, *args) -> List[base_component.BaseComponent]:
    dag = []
    logging.info("NitroML benchmarks:")
    seen = set()
    for repeatable_pipeline in self._pipelines:
      logging.info("\t%s", repeatable_pipeline.benchmark_name)
      logging.info("\t\tRUNNING")
      components = repeatable_pipeline.pipeline
      for component in components:
        if component in seen:
          continue
        # pylint: disable=protected-access
        component._instance_name = _qualified_name(
            component._instance_name, repeatable_pipeline.benchmark_name)
        # pylint: enable=protected-access
        seen.add(component)
      dag += components
    return dag


class BenchmarkResult(object):
  """Holder for benchmark result information.

  Benchmark results are automatically managed by the Benchmark class, and do
  not need to be explicitly manipulated by benchmark authors.
  """

  def __init__(self):
    self.pipelines = []


class Benchmark(abc.ABC):
  """A benchmark which can be composed of several benchmark methods.

  The Benchmark object design is inspired by `unittest.TestCase`.

  A benchmark file can contain multiple Benchmark subclasses to compose a suite
  of benchmarks.
  """

  def __init__(self):
    self._benchmark = self  # The sub-benchmark stack.
    self._result = None
    self._seen_benchmarks = None

  @abc.abstractmethod
  def benchmark(self):
    """Benchmark method to be overridden by subclasses."""

  @contextlib.contextmanager
  def sub_benchmark(self, name: Text):
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

  def evaluate(
      self,
      pipeline: List[base_component.BaseComponent],
      examples: types.Channel,
      model: types.Channel,
  ) -> None:
    """Adds a benchmark subgraph to the benchmark suite's workflow DAG.

    Automatically appends a TFX Evaluator component to the given DAG in order
    to benchmark the given `model` on `examples`.

    Args:
      pipeline: List of TFX components of the workflow DAG to benchmark.
      examples: An `standard_artifacts.Examples` Channel, usually produced by an
        ExampleGen component. Input to the benchmark Evaluator. Will use the
        'eval' key examples as the test dataset.
      model: A `standard_artifacts.Model` Channel, usually produced by a Trainer
        component. Input to the benchmark Evaluator.
    """

    # Strip common parts of benchmark names from benchmark ID.
    benchmark_name = self._benchmark.id()
    if benchmark_name in self._seen_benchmarks:
      raise ValueError("evaluate was already called once for this benchmark. "
                       "Consider creating a sub-benchmark instead.")
    self._seen_benchmarks.add(benchmark_name)

    # Automatically add an Evaluator component to evaluate the produced model on
    # the test set.
    # TODO(b/146611976): Include a Model-agnostic Evaluator which computes
    # metrics according to task type.
    evaluator = tfx.Evaluator(examples, model)
    self._result.pipelines.append(
        _BenchmarkPipeline(benchmark_name, pipeline, evaluator))

  def id(self):
    """The unique ID of this benchmark."""

    return f"{self.__class__.__qualname__}.benchmark"

  def __call__(self, *args):
    result = BenchmarkResult()
    self._seen_benchmarks = set()
    self._result = result
    try:
      self.benchmark()
    finally:
      self._result = None
      self._seen_benchmarks = set()
    return result


class _SubBenchmark(Benchmark):
  """A benchmark nested within another benchmark.

  SubBenchmarks allow pipelines to split creating a tree-like workflow where
  each leaf node is evaluated independently. This allows artifacts to be shared
  between benchmarks for maximum resource efficiency.
  """

  def __init__(self, parent: Benchmark, name: Text):
    super(_SubBenchmark, self).__init__()
    self._parent = parent
    self._name = name

  def benchmark(self):
    # Implement abstract method so that _SubBenchmarks can be instantiated
    # directly.
    raise RuntimeError("Sub-benchmarks are not intended to be called directly.")

  def id(self):
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


def run(benchmarks: List[Benchmark]):
  """Runs the given benchmarks using MyOrchestrator.

  First it concatenates all the benchmark pipelines into a single DAG
  metapipeline. Next it executes the workflow via my_orchestrator.run().

  When the `match` flag is set, matched benchmarks are filtered by name.

  Args:
    benchmarks: List of Benchmark instances to include in the suite.
  """

  if "runs_per_benchmark" in pipeline_args():
    runs_per_benchmark = int(pipeline_args()["runs_per_benchmark"])
    logging.info(
        "Using runs_per_benchmark=%d as specified "
        "in the config's pipeline_args.", runs_per_benchmark)
  else:
    runs_per_benchmark = FLAGS.runs_per_benchmark
    logging.info(
        "Using runs_per_benchmark=%d as specified "
        "by the --runs_per_benchmark flag.", runs_per_benchmark)

  if runs_per_benchmark <= 0:
    raise ValueError("runs_per_benchmark must be strictly positive; "
                     f"got runs_per_benchmark={runs_per_benchmark} instead.")

  pipelines = []
  for b in benchmarks:
    for benchmark_run in range(runs_per_benchmark):
      result = b()
      for pipeline in result.pipelines:
        if re.match(FLAGS.match, pipeline.benchmark_name):
          pipelines.append(
              _RepeatablePipeline(
                  pipeline,
                  repetition=benchmark_run + 1,  # One-index runs.
                  num_repetitions=runs_per_benchmark))
  my_orchestrator.run(_ConcatenatedPipeline(pipelines))


def main(*args, **kwargs):
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

  del args, kwargs  # Unused

  app.run(lambda _: run(_load_benchmarks()))
