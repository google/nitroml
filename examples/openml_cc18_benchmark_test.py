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

from nitroml.benchmark import results
from nitroml.benchmark.suites import testing_utils
from examples import openml_cc18_benchmark
from nitroml.testing import e2etest
import requests_mock

from ml_metadata import metadata_store


class OpenMLCC18BenchmarkTest(e2etest.BenchmarkTestCase):

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
      dataset_id_list = [1, 2]
      testing_utils.register_mock_urls(mocker, dataset_id_list)
      self.run_benchmarks(
          [openml_cc18_benchmark.OpenMLCC18Benchmark()],
          data_dir=os.path.join(self.pipeline_root, 'openML_mock_data'),
          mock_data=True,
          use_keras=use_keras,
          enable_tuning=enable_tuning,
      )

    instance_names = []
    for did in dataset_id_list:
      instance_name = '.'.join(
          ['OpenMLCC18Benchmark', 'benchmark', f'OpenML.mockdata_{did}'])
      instance_names.append(instance_name)

      if enable_tuning:
        self.assertComponentExecutionCount(8 * len(dataset_id_list))
        self.assertComponentSucceeded('.'.join(
            ['Tuner.AutoTrainer', instance_name]))
      else:
        self.assertComponentExecutionCount(7 * len(dataset_id_list))
      self.assertComponentSucceeded('.'.join(
          [f'CsvExampleGen.OpenML.mockdata_{did}', instance_name]))
      self.assertComponentSucceeded('.'.join(
          ['SchemaGen.AutoData', instance_name]))
      self.assertComponentSucceeded('.'.join(
          ['StatisticsGen.AutoData', instance_name]))
      self.assertComponentSucceeded('.'.join(
          ['Transform.AutoData', instance_name]))
      self.assertComponentSucceeded('.'.join(
          ['Trainer.AutoTrainer', instance_name]))
      self.assertComponentSucceeded('.'.join(['Evaluator.model',
                                              instance_name]))
      self.assertComponentSucceeded('.'.join(
          ['BenchmarkResultPublisher.model', instance_name]))

    # Load benchmark results.
    store = metadata_store.MetadataStore(self.metadata_config)
    df = results.overview(store)

    # Check benchmark results overview values.
    self.assertEqual(len(df.index), len(dataset_id_list))
    self.assertContainsSubset([
        'run_id',
        'benchmark_fullname',
        'benchmark',
        'run',
        'num_runs',
        'accuracy',
        'average_loss',
        'example_count',
        'weighted_example_count',
    ], df.columns.values.tolist())
    self.assertSameElements([1], df['run'].tolist())
    self.assertSameElements([1], df['num_runs'].tolist())
    self.assertSameElements(instance_names, df.benchmark.unique())


if __name__ == '__main__':
  e2etest.main()
