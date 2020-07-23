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
"""Tests for nitroml.examples.openml_cc18_benchmark.

  This test assumes that the openml_cc18 datasets exist.
"""

import os

from nitroml import results
from examples import openml_cc18_benchmark
from nitroml.suites import testing_utils
from nitroml.testing import e2etest
import requests_mock

from ml_metadata import metadata_store


class OpenMLCC18BenchmarkTest(e2etest.TestCase):

  def setUp(self):
    super(OpenMLCC18BenchmarkTest, self).setUp('nitroml_openml_cc_18_benchmark')

  @e2etest.parameterized.named_parameters(
      {
          'testcase_name': 'keras_trainer',
          'use_keras': True,
          'enable_tuning': False,
      }, {
          'testcase_name': 'keras_tuning_trainer',
          'use_keras': True,
          'enable_tuning': True,
      }, {
          'testcase_name': 'estimator_trainer',
          'use_keras': False,
          'enable_tuning': False,
      })
  def test(self, use_keras, enable_tuning):
    with requests_mock.Mocker() as mocker:
      testing_utils.register_mock_urls(mocker)
      self.run_benchmarks(
          [openml_cc18_benchmark.OpenMLCC18Benchmark()],
          data_dir=os.path.join(self.pipeline_root, 'openML_mock_data'),
          mock_data=True,
          use_keras=use_keras,
          enable_tuning=enable_tuning,
      )

    instance_name = '.'.join(['OpenMLCC18Benchmark', 'benchmark', 'mockdata'])
    if enable_tuning:
      self.assertComponentExecutionCount(8)
      self.assertComponentSucceeded('.'.join(['Tuner', instance_name]))
    else:
      self.assertComponentExecutionCount(7)
    self.assertComponentSucceeded('.'.join(['CsvExampleGen', instance_name]))
    self.assertComponentSucceeded('.'.join(['SchemaGen', instance_name]))
    self.assertComponentSucceeded('.'.join(['StatisticsGen', instance_name]))
    self.assertComponentSucceeded('.'.join(['Transform', instance_name]))
    self.assertComponentSucceeded('.'.join(['Trainer', instance_name]))
    self.assertComponentSucceeded('.'.join(['Evaluator', instance_name]))
    self.assertComponentSucceeded('.'.join(
        ['BenchmarkResultPublisher', instance_name]))

    # Load benchmark results.
    store = metadata_store.MetadataStore(self.metadata_config)
    df = results.overview(store)

    # Check benchmark results overview values.
    self.assertEqual(len(df.index), 1)
    self.assertContainsSubset([
        'benchmark',
        'run',
        'num_runs',
        'accuracy',
        'average_loss',
        'post_export_metrics/example_count',
    ], df.columns.values.tolist())
    self.assertSameElements([1], df['run'].tolist())
    self.assertSameElements([1], df['num_runs'].tolist())
    self.assertSameElements([instance_name], df.benchmark.unique())


if __name__ == '__main__':
  e2etest.main()
