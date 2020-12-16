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
"""Tests for nitroml.benchmark.result_publisher.component."""

from absl.testing import absltest

from nitroml.benchmark.result_publisher.component import BenchmarkResultPublisher
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(absltest.TestCase):

  def testMissingBenchmarkResultConstruction(self):
    publisher = BenchmarkResultPublisher(
        'test',
        channel_utils.as_channel([standard_artifacts.ModelEvaluation()]),
        run=1,
        num_runs=2)

    self.assertEqual('NitroML.BenchmarkResult',
                     publisher.outputs['benchmark_result'].type_name)

  def testInvalidBenchmarkNameThrows(self):
    with self.assertRaises(ValueError):
      BenchmarkResultPublisher(
          '',
          channel_utils.as_channel([standard_artifacts.ModelEvaluation()]),
          run=1,
          num_runs=2)

  def testContextProperties(self):
    publisher = BenchmarkResultPublisher(
        'test',
        channel_utils.as_channel([standard_artifacts.ModelEvaluation()]),
        run=1,
        num_runs=2,
        additional_context={
            'test_int': 1,
            'test_str': 'str',
            'test_float': 0.1
        })
    want = '{"test_float": 0.1, "test_int": 1, "test_str": "str"}'
    got = publisher.exec_properties['additional_context']
    self.assertEqual(want, got)


if __name__ == '__main__':
  absltest.main()
