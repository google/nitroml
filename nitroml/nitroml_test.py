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
"""Tests for nitroml.py."""

import abc

import types
from typing import List, Optional

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from nitroml import nitroml
from nitroml.benchmark.task import BenchmarkTask
from nitroml.subpipeline import Subpipeline
from nitroml.subpipeline import SubpipelineOutputs

from tfx import types as tfx_types
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.dsl.components.base.base_driver import BaseDriver
from tfx.dsl.components.base.executor_spec import ExecutorSpec
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.pipeline import Pipeline
from tfx.types import channel_utils
from tfx.types import standard_artifacts

from google.protobuf import message
from nitroml.protos import problem_statement_pb2 as ps_pb2

FLAGS = flags.FLAGS


class FakeExecutorSpec(ExecutorSpec):
  """A fake ExecutorSpec component for testing."""

  def encode(
      self,
      component_spec: Optional[tfx_types.ComponentSpec] = None
  ) -> message.Message:
    # Return an arbitrary proto.
    return ps_pb2.ProblemStatement()


class FakeExampleGen(BaseComponent):
  """A fake ExampleGen component for testing."""

  SPEC_CLASS = 'ExampleGenClass'
  EXECUTOR_SPEC = 'ExampleGenExecutorSpec'
  executor_spec = FakeExecutorSpec()
  driver_class = BaseDriver
  platform_config = None

  def __init__(self, instance_name: str = ''):
    self.examples = channel_utils.as_channel([standard_artifacts.Examples()])
    self.spec = types.SimpleNamespace(
        outputs=self.outputs, inputs={}, exec_properties={})
    self._downstream_nodes = set()
    self._upstream_nodes = set()
    self._instance_name = instance_name
    self._id = None

  @property
  def outputs(self):
    return SubpipelineOutputs({'examples': self.examples})


class FakeTrainer(BaseComponent):
  """A fake Trainer component for testing."""

  SPEC_CLASS = 'TrainerSpecClass'
  EXECUTOR_SPEC = 'TrainerExecutorSpec'
  executor_spec = FakeExecutorSpec()
  driver_class = BaseDriver
  platform_config = None

  def __init__(self, instance_name: str = ''):
    self.model = channel_utils.as_channel([standard_artifacts.Model()])
    self.spec = types.SimpleNamespace(
        outputs=self.outputs, inputs={}, exec_properties={})
    self._downstream_nodes = set()
    self._upstream_nodes = set()
    self._instance_name = instance_name
    self._id = None

  @property
  def outputs(self):
    return SubpipelineOutputs({'model': self.model})


class FakeSubpipeline(Subpipeline):
  """A fake Subpipeline definition for testing."""

  def __init__(self, instance_name: str = ''):
    # Intentionally skip calling super. Goal is to match the parent type only.
    self.trainer = FakeTrainer(instance_name)

  @property
  def id(self) -> str:
    return 'FakeSubpipeline'

  @property
  def components(self) -> List[BaseComponent]:
    return [self.trainer]

  @property
  def model(self) -> tfx_types.Channel:
    return self.trainer.outputs.model

  @property
  def outputs(self) -> SubpipelineOutputs:
    return SubpipelineOutputs({'model': self.model})


class FakePipeline(Pipeline):
  """A fake TFX Pipeline definition for testing."""

  def __init__(self):
    # Intentionally skip calling super. Goal is to match the parent type only.
    self.example_gen = FakeExampleGen()
    self.trainer = FakeTrainer()

  @property
  def components(self) -> List[BaseComponent]:
    return [self.example_gen, self.trainer]


class FakeBeamDagRunner(beam_dag_runner.BeamDagRunner):
  """A fake Beam TFX runner for testing."""

  def __init__(self):
    super().__init__()
    self._got_pipeline = []

  @property
  def got_pipeline(self):
    return self._got_pipeline

  def run(self, pipeline):
    self._got_pipeline = pipeline
    return pipeline


class FakeBenchmarkTask(BenchmarkTask):

  def __init__(self):
    self._example_gen = FakeExampleGen()

  @property
  def name(self) -> str:
    """Returns the task's name."""

    return 'FakeBenchmarkTask'

  @property
  def components(self) -> List[BaseComponent]:
    """Returns TFX components required for the task."""

    return [self._example_gen]

  @property
  def train_and_eval_examples(self) -> tfx_types.Channel:
    """Returns train and eval labeled examples."""

    return self._example_gen.outputs.examples

  @property
  def test_examples(self) -> tfx_types.Channel:
    """Returns test labeled examples."""

    return self._example_gen.outputs.examples

  @property
  def test_example_splits(self) -> List[str]:
    return ['test']

  @property
  def problem_statement(self) -> ps_pb2.ProblemStatement:
    """Returns the ProblemStatement associated with this BenchmarkTask."""

    return ps_pb2.ProblemStatement(
        owner=['nitroml'],
        tasks=[
            ps_pb2.Task(
                name='Test',
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='test')),
            )
        ])




class Benchmarks:
  """Hide these benchmarks from the nitroml.main runner below."""

  class Benchmark1(nitroml.Benchmark):

    def test_this_benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      self.evaluate(task=task, model=pipeline.model)

    def _benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      self.evaluate(task=task, model=pipeline.model)

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      self.evaluate(task=task, model=pipeline.model)

    def benchmark_my_pipeline(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      self.evaluate(task=task, model=pipeline.model)

  class BenchmarkComponents(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      trainer = self.add(FakeTrainer())
      self.evaluate(task=task, model=trainer.outputs.model)

  class BenchmarkSubpipeline(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      self.evaluate(task=task, model=pipeline.model)

  class BenchmarkTuple(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      _, trainer = self.add((task.components, FakeTrainer()))
      self.evaluate(task=task, model=trainer.outputs.model)

  class BenchmarkList(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      _, trainer = self.add([task.components, FakeTrainer()])
      self.evaluate(task=task, model=trainer.outputs.model)

  class BenchmarkDict(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      components = self.add({
          'example_gen': FakeExampleGen(),
          'trainer': FakeTrainer()
      })
      self.evaluate(task=task, model=components['trainer'].outputs.model)

  class BenchmarkDeeplyNested(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      components = self.add({
          'components': [task.components, {
              'trainer': FakeTrainer()
          }],
      })
      self.evaluate(
          task=task, model=components['components'][1]['trainer'].outputs.model)

  class SubBenchmark(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      with self.sub_benchmark('one'):
        self.evaluate(task=task, model=pipeline.model)

  class BenchmarkAndSubBenchmark(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      self.evaluate(task=task, model=pipeline.model)
      with self.sub_benchmark('one'):
        self.evaluate(task=task, model=pipeline.model)

  class NestedSubBenchmarks(nitroml.Benchmark):

    def benchmark(self):
      with self.sub_benchmark('one'):
        with self.sub_benchmark('two'):
          task = FakeBenchmarkTask()
          self.add(task.components)
          pipeline = self.add(FakeSubpipeline())
          self.evaluate(task=task, model=pipeline.model)

  class SharedNestedSubBenchmarks(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      with self.sub_benchmark('one'):
        with self.sub_benchmark('two'):
          self.evaluate(task=task, model=pipeline.model)

  class MultipleSubBenchmarks(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      with self.sub_benchmark('mnist'):
        self.evaluate(task=task, model=pipeline.model)
      with self.sub_benchmark('chicago_taxi'):
        self.evaluate(task=task, model=pipeline.model)

  # Partial run benchmarks below:

  class PartialRunSkipBenchmark(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      trainer = self.add(FakeTrainer(), skip=True)
      self.evaluate(task=task, model=trainer.outputs.model, skip=True)

  class PartialRunOnlyBenchmark(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components, only=True)
      trainer = self.add(FakeTrainer())
      self.evaluate(task=task, model=trainer.outputs.model)

  # Error causing benchmarks below:

  class CallAddBenchmarkTwice(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      self.evaluate(task=task, model=pipeline.model)
      self.evaluate(task=task, model=pipeline.model)

  class CallAddBenchmarkTwiceInSubBenchmark(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      with self.sub_benchmark('one'):
        self.evaluate(task=task, model=pipeline.model)
        self.evaluate(task=task, model=pipeline.model)

  class TwoSubBenchmarksWithSameName(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components)
      pipeline = self.add(FakeSubpipeline())
      with self.sub_benchmark('one'):
        self.evaluate(task=task, model=pipeline.model)
      with self.sub_benchmark('one'):
        self.evaluate(task=task, model=pipeline.model)

  class BenchmarkSubpipelineWithoutAlways(nitroml.Benchmark):

    def benchmark(self):
      self.add(FakeSubpipeline(), always=False)

  class PartialRunSkipAndOnlyBenchmark(nitroml.Benchmark):

    def benchmark(self):
      task = FakeBenchmarkTask()
      self.add(task.components, skip=True, only=True)
      trainer = self.add(FakeTrainer())
      self.evaluate(task=task, model=trainer.outputs.model)


class NitroMLTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name':
              'default',
          'runs_per_benchmark_flag':
              1,
          'benchmarks': [Benchmarks.BenchmarkSubpipeline()],
          'want_benchmarks': ['Benchmarks.BenchmarkSubpipeline.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkSubpipeline.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkSubpipeline.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkSubpipeline.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkSubpipeline.benchmark',
          ]
      }, {
          'testcase_name':
              'runs_per_benchmark_flag_set',
          'runs_per_benchmark_flag':
              3,
          'benchmarks': [Benchmarks.BenchmarkSubpipeline()],
          'want_benchmarks': [
              'Benchmarks.BenchmarkSubpipeline.benchmark.run_1_of_3',
              'Benchmarks.BenchmarkSubpipeline.benchmark.run_2_of_3',
              'Benchmarks.BenchmarkSubpipeline.benchmark.run_3_of_3',
          ],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkSubpipeline.benchmark.run_1_of_3',
              'FakeTrainer.Benchmarks.BenchmarkSubpipeline.benchmark.run_1_of_3',
              'Evaluator.model.Benchmarks.BenchmarkSubpipeline.benchmark.run_1_of_3',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkSubpipeline.benchmark.run_1_of_3',
              'FakeExampleGen.Benchmarks.BenchmarkSubpipeline.benchmark.run_2_of_3',
              'FakeTrainer.Benchmarks.BenchmarkSubpipeline.benchmark.run_2_of_3',
              'Evaluator.model.Benchmarks.BenchmarkSubpipeline.benchmark.run_2_of_3',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkSubpipeline.benchmark.run_2_of_3',
              'FakeExampleGen.Benchmarks.BenchmarkSubpipeline.benchmark.run_3_of_3',
              'FakeTrainer.Benchmarks.BenchmarkSubpipeline.benchmark.run_3_of_3',
              'Evaluator.model.Benchmarks.BenchmarkSubpipeline.benchmark.run_3_of_3',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkSubpipeline.benchmark.run_3_of_3',
          ]
      })
  @flagsaver.flagsaver
  def test_runs_per_benchmark(self, runs_per_benchmark_flag, benchmarks,
                              want_benchmarks, want_components):
    FLAGS.runs_per_benchmark = runs_per_benchmark_flag
    pipeline = nitroml.run(benchmarks, tfx_runner=FakeBeamDagRunner())
    self.assertEqual(want_benchmarks, pipeline.benchmark_names)
    self.assertCountEqual(want_components, [c.id for c in pipeline.components])

  @parameterized.named_parameters(
      {
          'testcase_name': 'zero_runs_per_benchmark_flag',
          'runs_per_benchmark_flag': 0,
          'benchmarks': [Benchmarks.BenchmarkSubpipeline()],
      }, {
          'testcase_name': 'negative_runs_per_benchmark_flag',
          'runs_per_benchmark_flag': -1,
          'benchmarks': [Benchmarks.BenchmarkSubpipeline()],
      })
  @flagsaver.flagsaver
  def test_run_per_benchmark_errors(self, runs_per_benchmark_flag, benchmarks):
    FLAGS.runs_per_benchmark = runs_per_benchmark_flag
    with self.assertRaises(ValueError):
      nitroml.run(benchmarks, tfx_runner=FakeBeamDagRunner())


  @parameterized.named_parameters(
      {
          'testcase_name': 'none_name',
          'prefix': None,
          'name': 'test',
          'want': 'test',
      }, {
          'testcase_name': 'empty_name',
          'prefix': '',
          'name': 'test',
          'want': 'test',
      }, {
          'testcase_name': 'existing_name',
          'prefix': 'foo',
          'name': 'test',
          'want': 'foo.test',
      }, {
          'testcase_name': 'num_runs_is_2',
          'prefix': 'foo',
          'name': 'test',
          'want': 'foo.test.run_1_of_3',
          'num_runs': 3,
      })
  def test_qualified_name(self, prefix, name, want, run=1, num_runs=1):
    self.assertEqual(want, nitroml._qualified_name(prefix, name, run, num_runs))

  @parameterized.named_parameters(
      {
          'testcase_name':
              'components',
          'benchmark':
              Benchmarks.BenchmarkComponents(),
          'want_benchmarks': ['Benchmarks.BenchmarkComponents.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkComponents.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkComponents.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkComponents.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkComponents.benchmark',
          ],
      }, {
          'testcase_name':
              'subpipeline',
          'benchmark':
              Benchmarks.BenchmarkSubpipeline(),
          'want_benchmarks': ['Benchmarks.BenchmarkSubpipeline.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkSubpipeline.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkSubpipeline.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkSubpipeline.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkSubpipeline.benchmark',
          ],
      }, {
          'testcase_name':
              'components_tuple',
          'benchmark':
              Benchmarks.BenchmarkTuple(),
          'want_benchmarks': ['Benchmarks.BenchmarkTuple.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkTuple.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkTuple.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkTuple.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkTuple.benchmark',
          ],
      }, {
          'testcase_name':
              'components_list',
          'benchmark':
              Benchmarks.BenchmarkList(),
          'want_benchmarks': ['Benchmarks.BenchmarkList.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkList.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkList.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkList.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkList.benchmark',
          ],
      }, {
          'testcase_name':
              'components_dict',
          'benchmark':
              Benchmarks.BenchmarkDict(),
          'want_benchmarks': ['Benchmarks.BenchmarkDict.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkDict.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkDict.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkDict.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkDict.benchmark',
          ],
      }, {
          'testcase_name':
              'components_deeply_nested',
          'benchmark':
              Benchmarks.BenchmarkDeeplyNested(),
          'want_benchmarks': ['Benchmarks.BenchmarkDeeplyNested.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkDeeplyNested.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkDeeplyNested.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkDeeplyNested.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkDeeplyNested'
              '.benchmark',
          ],
      }, {
          'testcase_name':
              'sub_benchmark',
          'benchmark':
              Benchmarks.SubBenchmark(),
          'want_benchmarks': ['Benchmarks.SubBenchmark.benchmark.one'],
          'want_components': [
              'FakeExampleGen.Benchmarks.SubBenchmark.benchmark',
              'FakeTrainer.Benchmarks.SubBenchmark.benchmark',
              'Evaluator.model.Benchmarks.SubBenchmark.benchmark.one',
              'BenchmarkResultPublisher.model.Benchmarks.SubBenchmark.benchmark.one',
          ]
      }, {
          'testcase_name':
              'benchmark_and_sub_benchmark',
          'benchmark':
              Benchmarks.BenchmarkAndSubBenchmark(),
          'want_benchmarks': [
              'Benchmarks.BenchmarkAndSubBenchmark.benchmark',
              'Benchmarks.BenchmarkAndSubBenchmark.benchmark.one'
          ],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkAndSubBenchmark.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkAndSubBenchmark.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkAndSubBenchmark.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkAndSubBenchmark'
              '.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkAndSubBenchmark.benchmark.one',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkAndSubBenchmark'
              '.benchmark.one',
          ]
      }, {
          'testcase_name':
              'nested_sub_benchmark',
          'benchmark':
              Benchmarks.NestedSubBenchmarks(),
          'want_benchmarks':
              ['Benchmarks.NestedSubBenchmarks.benchmark.one.two'],
          'want_components': [
              'FakeExampleGen.Benchmarks.NestedSubBenchmarks.benchmark.one.two',
              'FakeTrainer.Benchmarks.NestedSubBenchmarks.benchmark.one.two',
              'Evaluator.model.Benchmarks.NestedSubBenchmarks.benchmark.one.two',
              'BenchmarkResultPublisher.model.Benchmarks.NestedSubBenchmarks'
              '.benchmark.one.two',
          ]
      }, {
          'testcase_name':
              'shared_nested_sub_benchmark',
          'benchmark':
              Benchmarks.SharedNestedSubBenchmarks(),
          'want_benchmarks':
              ['Benchmarks.SharedNestedSubBenchmarks.benchmark.one.two'],
          'want_components': [
              'FakeExampleGen.Benchmarks.SharedNestedSubBenchmarks.benchmark',
              'FakeTrainer.Benchmarks.SharedNestedSubBenchmarks.benchmark',
              'Evaluator.model.Benchmarks.SharedNestedSubBenchmarks.benchmark.one'
              '.two',
              'BenchmarkResultPublisher.model.Benchmarks.SharedNestedSubBenchmarks'
              '.benchmark.one.two',
          ]
      }, {
          'testcase_name':
              'multiple_sub_benchmarks',
          'benchmark':
              Benchmarks.MultipleSubBenchmarks(),
          'want_benchmarks': [
              'Benchmarks.MultipleSubBenchmarks.benchmark.chicago_taxi',
              'Benchmarks.MultipleSubBenchmarks.benchmark.mnist'
          ],
          'want_components': [
              'FakeExampleGen.Benchmarks.MultipleSubBenchmarks.benchmark',
              'FakeTrainer.Benchmarks.MultipleSubBenchmarks.benchmark',
              'Evaluator.model.Benchmarks.MultipleSubBenchmarks.benchmark'
              '.chicago_taxi',
              'BenchmarkResultPublisher.model.Benchmarks.MultipleSubBenchmarks'
              '.benchmark.chicago_taxi',
              'Evaluator.model.Benchmarks.MultipleSubBenchmarks.benchmark'
              '.mnist',
              'BenchmarkResultPublisher.model.Benchmarks.MultipleSubBenchmarks'
              '.benchmark.mnist',
          ]
      })
  def test_run(self, benchmark, want_benchmarks, want_components):
    pipeline = nitroml.run([benchmark], tfx_runner=FakeBeamDagRunner())
    self.assertCountEqual(want_benchmarks, pipeline.benchmark_names)
    self.assertCountEqual(want_components, [c.id for c in pipeline.components])

  @parameterized.named_parameters(
      {
          'testcase_name':
              'none',
          'benchmark':
              Benchmarks.BenchmarkComponents(),
          'want_benchmarks': ['Benchmarks.BenchmarkComponents.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.BenchmarkComponents.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkComponents.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkComponents.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkComponents'
              '.benchmark',
          ],
          'want_partial_run_components': [
              'FakeExampleGen.Benchmarks.BenchmarkComponents.benchmark',
              'FakeTrainer.Benchmarks.BenchmarkComponents.benchmark',
              'Evaluator.model.Benchmarks.BenchmarkComponents.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.BenchmarkComponents'
              '.benchmark',
          ],
      }, {
          'testcase_name':
              'skip',
          'benchmark':
              Benchmarks.PartialRunSkipBenchmark(),
          'want_benchmarks': ['Benchmarks.PartialRunSkipBenchmark.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.PartialRunSkipBenchmark.benchmark',
              'FakeTrainer.Benchmarks.PartialRunSkipBenchmark.benchmark',
              'Evaluator.model.Benchmarks.PartialRunSkipBenchmark.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.PartialRunSkipBenchmark.benchmark',
          ],
          'want_partial_run_components':
              ['FakeExampleGen.Benchmarks.PartialRunSkipBenchmark.benchmark',],
      }, {
          'testcase_name':
              'only',
          'benchmark':
              Benchmarks.PartialRunOnlyBenchmark(),
          'want_benchmarks': ['Benchmarks.PartialRunOnlyBenchmark.benchmark'],
          'want_components': [
              'FakeExampleGen.Benchmarks.PartialRunOnlyBenchmark.benchmark',
              'FakeTrainer.Benchmarks.PartialRunOnlyBenchmark.benchmark',
              'Evaluator.model.Benchmarks.PartialRunOnlyBenchmark.benchmark',
              'BenchmarkResultPublisher.model.Benchmarks.PartialRunOnlyBenchmark.benchmark',
          ],
          'want_partial_run_components':
              ['FakeExampleGen.Benchmarks.PartialRunOnlyBenchmark.benchmark',],
      })
  def test_partial_run(self, benchmark, want_benchmarks, want_components,
                       want_partial_run_components):
    tfx_runner = FakeBeamDagRunner()
    pipeline = nitroml.run([benchmark], tfx_runner=tfx_runner)
    self.assertCountEqual(want_benchmarks, pipeline.benchmark_names)
    self.assertCountEqual(want_components, [c.id for c in pipeline.components])
    self.assertSameElements(
        want_partial_run_components,
        [n.pipeline_node.node_info.id for n in tfx_runner.got_pipeline.nodes])  # pytype: disable=attribute-error

  @parameterized.named_parameters(
      {
          'testcase_name': 'call_add_benchmark_twice',
          'benchmark': Benchmarks.CallAddBenchmarkTwice(),
      }, {
          'testcase_name': 'call_add_benchmark_twice_in_sub-benchmark',
          'benchmark': Benchmarks.CallAddBenchmarkTwiceInSubBenchmark(),
      }, {
          'testcase_name': 'call_two_subbenchmarks_with_same_name',
          'benchmark': Benchmarks.TwoSubBenchmarksWithSameName(),
      }, {
          'testcase_name': 'call_run_on_subpipeline_without_setting_always',
          'benchmark': Benchmarks.BenchmarkSubpipelineWithoutAlways(),
      }, {
          'testcase_name': 'partial_run_with_skip_and_only',
          'benchmark': Benchmarks.PartialRunSkipAndOnlyBenchmark(),
      })
  def test_run_error(self, benchmark):
    with self.assertRaises(ValueError):
      nitroml.run([benchmark], tfx_runner=FakeBeamDagRunner())

  @parameterized.named_parameters(
      {
          'testcase_name':
              'filter_by_mnist',
          'match':
              '.*mnist.*',
          'benchmark':
              Benchmarks.MultipleSubBenchmarks(),
          'want_benchmarks':
              ['Benchmarks.MultipleSubBenchmarks.benchmark.mnist'],
          'want_components': [
              'FakeExampleGen.Benchmarks.MultipleSubBenchmarks.benchmark',
              'FakeTrainer.Benchmarks.MultipleSubBenchmarks.benchmark',
              'Evaluator.model.Benchmarks.MultipleSubBenchmarks.benchmark'
              '.mnist',
              'BenchmarkResultPublisher.model.Benchmarks.MultipleSubBenchmarks'
              '.benchmark.mnist',
          ]
      }, {
          'testcase_name':
              'filter_by_taxi',
          'match':
              '.*taxi.*',
          'benchmark':
              Benchmarks.MultipleSubBenchmarks(),
          'want_benchmarks':
              ['Benchmarks.MultipleSubBenchmarks.benchmark.chicago_taxi'],
          'want_components': [
              'FakeExampleGen.Benchmarks.MultipleSubBenchmarks.benchmark',
              'FakeTrainer.Benchmarks.MultipleSubBenchmarks.benchmark',
              'Evaluator.model.Benchmarks.MultipleSubBenchmarks.benchmark'
              '.chicago_taxi',
              'BenchmarkResultPublisher.model.Benchmarks.MultipleSubBenchmarks'
              '.benchmark.chicago_taxi',
          ]
      })
  @flagsaver.flagsaver
  def test_benchmark_match(self, match, benchmark, want_benchmarks,
                           want_components):
    FLAGS.match = match
    pipeline = nitroml.run([benchmark], tfx_runner=FakeBeamDagRunner())
    self.assertEqual(want_benchmarks, pipeline.benchmark_names)
    self.assertCountEqual(want_components, [c.id for c in pipeline.components])

  @parameterized.named_parameters({
      'testcase_name': 'filter_by_invalid',
      'match': '.*invalid.*',
      'benchmark': Benchmarks.MultipleSubBenchmarks(),
  })
  @flagsaver.flagsaver
  def test_benchmark_match_error(self, match, benchmark):
    FLAGS.match = match
    with self.assertRaises(ValueError):
      nitroml.run([benchmark], tfx_runner=FakeBeamDagRunner())


class NitroMLGetSubclassesTest(absltest.TestCase):

  def testNoSubclass(self):

    class BaseClass:
      pass

    subclasses = nitroml._get_subclasses(BaseClass)
    self.assertEqual(subclasses, [])

  def testSubclassesLevel1(self):

    class BaseClass:
      pass

    class SubClass1(BaseClass):
      pass

    class SubClass2(BaseClass):
      pass

    subclasses = nitroml._get_subclasses(BaseClass)
    self.assertSameElements(subclasses, [SubClass1, SubClass2])

  def testSubclassesLevel2(self):

    class BaseClass:
      pass

    class SubClass1(BaseClass):
      pass

    class SubClass11(SubClass1):
      pass

    class SubClass12(SubClass1):
      pass

    class SubClass2(BaseClass):
      pass

    class SubClass21(SubClass2):
      pass

    subclasses = nitroml._get_subclasses(BaseClass)
    self.assertSameElements(
        subclasses, [SubClass1, SubClass2, SubClass11, SubClass12, SubClass21])

  def testAbstractSubclasses(self):

    class BaseClass(abc.ABC):
      pass

    class AbstractClass(BaseClass, abc.ABC):
      pass

    class SubClass1(AbstractClass):
      pass

    class SubClass2(AbstractClass):
      pass

    subclasses = nitroml._get_subclasses(BaseClass)
    self.assertSameElements(subclasses, [AbstractClass, SubClass1, SubClass2])


if __name__ == '__main__':
  absltest.main()
