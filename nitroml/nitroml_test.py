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
import json
import re
import sys

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from nitroml import nitroml

from tfx.orchestration.beam import beam_dag_runner
from tfx.types import channel_utils
from tfx.types import standard_artifacts

FLAGS = flags.FLAGS


class FakePipeline(object):
  """A fake pipeline definition for testing."""

  @property
  def components(self):
    return []

  @property
  def examples(self):
    examples = standard_artifacts.Examples()
    return channel_utils.as_channel([examples])

  @property
  def model(self):
    model = standard_artifacts.Model()
    return channel_utils.as_channel([model])


class FakeTfxRunner(beam_dag_runner.BeamDagRunner):
  """A fake TFX runner for testing."""

  def run(self, pipeline):
    return pipeline


class Benchmarks(object):
  """Hide these benchmarks from the nitroml.main runner below."""

  class Benchmark1(nitroml.Benchmark):

    def test_this_benchmark(self):
      pipeline = FakePipeline()
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)

    def _benchmark(self):
      pipeline = FakePipeline()
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)

    def benchmark(self):
      pipeline = FakePipeline()
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)

    def benchmark_my_pipeline(self):
      pipeline = FakePipeline()
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)

  class BenchmarkNoComponents(nitroml.Benchmark):

    def benchmark(self):
      pipeline = FakePipeline()
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)

  class SubBenchmark(nitroml.Benchmark):

    def benchmark(self):
      pipeline = FakePipeline()
      with self.sub_benchmark('one'):
        self.evaluate(
            pipeline.components,
            examples=pipeline.examples,
            model=pipeline.model)

  class BenchmarkAndSubBenchmark(nitroml.Benchmark):

    def benchmark(self):
      pipeline = FakePipeline()
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)
      with self.sub_benchmark('one'):
        self.evaluate(
            pipeline.components,
            examples=pipeline.examples,
            model=pipeline.model)

  class NestedSubBenchmarks(nitroml.Benchmark):

    def benchmark(self):
      pipeline = FakePipeline()
      with self.sub_benchmark('one'):
        with self.sub_benchmark('two'):
          self.evaluate(
              pipeline.components,
              examples=pipeline.examples,
              model=pipeline.model)

  class MultipleSubBenchmarks(nitroml.Benchmark):

    def benchmark(self):
      pipeline = FakePipeline()
      with self.sub_benchmark('mnist'):
        self.evaluate(
            pipeline.components,
            examples=pipeline.examples,
            model=pipeline.model)
      with self.sub_benchmark('chicago_taxi'):
        self.evaluate(
            pipeline.components,
            examples=pipeline.examples,
            model=pipeline.model)

  # Error causing benchmarks below:

  class CallAddBenchmarkTwice(nitroml.Benchmark):

    def benchmark(self):
      pipeline = FakePipeline()
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)
      self.evaluate(
          pipeline.components, examples=pipeline.examples, model=pipeline.model)

  class CallAddBenchmarkTwiceInSubBenchmark(nitroml.Benchmark):

    def benchmark(self):
      pipeline = FakePipeline()
      with self.sub_benchmark('one'):
        self.evaluate(
            pipeline.components,
            examples=pipeline.examples,
            model=pipeline.model)
        self.evaluate(
            pipeline.components,
            examples=pipeline.examples,
            model=pipeline.model)


class NitroMLTest(parameterized.TestCase, absltest.TestCase):

  def setUp(self):
    super(NitroMLTest, self).setUp()
    flags.FLAGS(sys.argv)
    # Reset flags.
    FLAGS.runs_per_benchmark = 1

  @parameterized.named_parameters(
      {
          'testcase_name': 'default',
          'runs_per_benchmark_flag': 1,
          'benchmarks': [Benchmarks.BenchmarkNoComponents()],
          'want_benchmarks': ['Benchmarks.BenchmarkNoComponents.benchmark']
      }, {
          'testcase_name':
              'runs_per_benchmark flag set',
          'runs_per_benchmark_flag':
              3,
          'benchmarks': [Benchmarks.BenchmarkNoComponents()],
          'want_benchmarks': [
              'Benchmarks.BenchmarkNoComponents.benchmark.run_1_of_3',
              'Benchmarks.BenchmarkNoComponents.benchmark.run_2_of_3',
              'Benchmarks.BenchmarkNoComponents.benchmark.run_3_of_3',
          ]
      })
  def test_run(self, runs_per_benchmark_flag, benchmarks, want_benchmarks):
    FLAGS.runs_per_benchmark = runs_per_benchmark_flag
    benchmark_names = nitroml.run(benchmarks, tfx_runner=FakeTfxRunner())
    self.assertEqual(want_benchmarks, benchmark_names)

  @parameterized.named_parameters(
      {
          'testcase_name': 'zero runs_per_benchmark flag',
          'runs_per_benchmark_flag': 0,
          'benchmarks': [Benchmarks.BenchmarkNoComponents()],
      }, {
          'testcase_name': 'negative runs_per_benchmark flag',
          'runs_per_benchmark_flag': -1,
          'benchmarks': [Benchmarks.BenchmarkNoComponents()],
      })
  def test_run_errors(self, runs_per_benchmark_flag, benchmarks):
    FLAGS.runs_per_benchmark = runs_per_benchmark_flag
    with self.assertRaises(ValueError):
      nitroml.run(benchmarks, tfx_runner=FakeTfxRunner())


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
      })
  def test_append_to_instance_name(self, prefix, name, want):
    self.assertEqual(want, nitroml._qualified_name(prefix, name))

  @parameterized.named_parameters(
      {
          'testcase_name': 'no components',
          'benchmark': Benchmarks.BenchmarkNoComponents(),
          'want_benchmarks': ['Benchmarks.BenchmarkNoComponents.benchmark']
      }, {
          'testcase_name': 'sub-benchmark',
          'benchmark': Benchmarks.SubBenchmark(),
          'want_benchmarks': ['Benchmarks.SubBenchmark.benchmark.one']
      }, {
          'testcase_name':
              'benchmark and sub-benchmark',
          'benchmark':
              Benchmarks.BenchmarkAndSubBenchmark(),
          'want_benchmarks': [
              'Benchmarks.BenchmarkAndSubBenchmark.benchmark',
              'Benchmarks.BenchmarkAndSubBenchmark.benchmark.one'
          ]
      }, {
          'testcase_name':
              'nested sub-benchmark',
          'benchmark':
              Benchmarks.NestedSubBenchmarks(),
          'want_benchmarks':
              ['Benchmarks.NestedSubBenchmarks.benchmark.one.two']
      }, {
          'testcase_name':
              'multiple sub-benchmarks',
          'benchmark':
              Benchmarks.MultipleSubBenchmarks(),
          'want_benchmarks': [
              'Benchmarks.MultipleSubBenchmarks.benchmark.chicago_taxi',
              'Benchmarks.MultipleSubBenchmarks.benchmark.mnist'
          ]
      })
  def test_evaluate(self, benchmark, want_benchmarks):
    result = benchmark()
    self.assertEqual(want_benchmarks,
                     sorted([b.benchmark_name for b in result.pipelines]))

  @parameterized.named_parameters(
      {
          'testcase_name': 'call add benchmark twice',
          'benchmark': Benchmarks.CallAddBenchmarkTwice(),
      }, {
          'testcase_name': 'call add benchmark twice in sub-benchmark',
          'benchmark': Benchmarks.CallAddBenchmarkTwiceInSubBenchmark(),
      })
  def test_evaluate_error(self, benchmark):
    with self.assertRaises(ValueError):
      benchmark()

  @parameterized.named_parameters(
      {
          'testcase_name':
              'filter by mnist',
          'filter_regex':
              '.*mnist.*',
          'benchmark':
              Benchmarks.MultipleSubBenchmarks(),
          'want_benchmarks':
              ['Benchmarks.MultipleSubBenchmarks.benchmark.mnist']
      }, {
          'testcase_name':
              'filter by taxi',
          'filter_regex':
              '.*taxi.*',
          'benchmark':
              Benchmarks.MultipleSubBenchmarks(),
          'want_benchmarks':
              ['Benchmarks.MultipleSubBenchmarks.benchmark.chicago_taxi']
      })
  def test_benchmark_filter(self, filter_regex, benchmark, want_benchmarks):
    result = benchmark()
    self.assertEqual(want_benchmarks, [
        b.benchmark_name
        for b in result.pipelines
        if re.match(filter_regex, b.benchmark_name)
    ])


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
