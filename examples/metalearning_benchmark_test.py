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
"""Tests for nitroml.examples.metalearning_benchmark."""

import os

from nitroml.analytics import mlmd_analytics
from nitroml.benchmark import results
from nitroml.benchmark.suites import testing_utils
from examples import metalearning_benchmark
from nitroml.testing import e2etest
import pandas as pd
import requests_mock

from ml_metadata import metadata_store


class MetaLearningTest(e2etest.BenchmarkTestCase):

  @e2etest.parameterized.named_parameters(
      {
          'testcase_name': 'metalearning_algo_majority-voting',
          'algorithm': 'majority_voting',
      },
      {
          'testcase_name': 'metalearning_algo_nearest-neighbor',
          'algorithm': 'nearest_neighbor',
      },
  )
  def test(self, algorithm):
    with requests_mock.Mocker() as mocker:
      testing_utils.register_mock_urls(mocker)
      self.run_benchmarks([metalearning_benchmark.MetaLearningBenchmark()],
                          data_dir=os.path.join(self.pipeline_root,
                                                'mock_metalearning_openml'),
                          mock_data=True,
                          algorithm=algorithm)

    instance_name = 'MetaLearningBenchmark.benchmark'
    self.assertComponentSucceeded(
        f'CsvExampleGen.OpenML.mockdata_1.{instance_name}')
    self.assertComponentSucceeded(
        f'AugmentedTuner.train.OpenML.mockdata_1.{instance_name}')
    self.assertComponentSucceeded(
        f'SchemaGen.AutoData.train.OpenML.mockdata_1.{instance_name}')
    self.assertComponentSucceeded(
        f'StatisticsGen.AutoData.train.OpenML.mockdata_1.{instance_name}')
    self.assertComponentSucceeded(
        f'Transform.AutoData.train.OpenML.mockdata_1.{instance_name}')

    instance_name_2 = 'MetaLearningBenchmark.benchmark.OpenML.mockdata_2'
    self.assertComponentSucceeded(
        f'CsvExampleGen.OpenML.mockdata_2.{instance_name_2}')
    self.assertComponentSucceeded(f'SchemaGen.AutoData.{instance_name_2}')
    self.assertComponentSucceeded(f'StatisticsGen.AutoData.{instance_name_2}')
    self.assertComponentSucceeded(f'Transform.AutoData.{instance_name_2}')
    self.assertComponentSucceeded(f'AugmentedTuner.{instance_name_2}')
    self.assertComponentSucceeded(f'Trainer.{instance_name_2}')
    self.assertComponentSucceeded(f'Evaluator.model.{instance_name_2}')
    self.assertComponentSucceeded(
        f'BenchmarkResultPublisher.model.{instance_name_2}')

    # Load benchmark results.
    store = metadata_store.MetadataStore(self.metadata_config)
    df = results.overview(store)
    # Check benchmark results overview values.
    self.assertEqual(len(df.index), 1)
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
    self.assertSameElements([instance_name_2], df.benchmark.unique())

    # TODO(b/178401753): Reimplement this test when name parsing is updated for
    # IR-based runners.
    # Load Analytics
    analytics = mlmd_analytics.Analytics(store=store)
    # Check test_pipeline run analysis
    run = analytics.get_latest_pipeline_run()
    self.assertEqual('test_pipeline', run.name)

    want_components = {
        f'CsvExampleGen.OpenML.mockdata_1.{instance_name}',
        f'AugmentedTuner.train.OpenML.mockdata_1.{instance_name}',
        f'SchemaGen.AutoData.train.OpenML.'
        f'mockdata_1.{instance_name}',
        f'StatisticsGen.AutoData.train.OpenML.'
        f'mockdata_1.{instance_name}',
        f'Transform.AutoData.train.OpenML.'
        f'mockdata_1.{instance_name}',
        f'CsvExampleGen.OpenML.mockdata_2.{instance_name_2}',
        f'SchemaGen.AutoData.{instance_name_2}',
        f'StatisticsGen.AutoData.{instance_name_2}',
        f'Transform.AutoData.{instance_name_2}',
        f'AugmentedTuner.{instance_name_2}',
        f'Trainer.{instance_name_2}',
        f'Evaluator.model.{instance_name_2}',
        f'BenchmarkResultPublisher.model.{instance_name_2}',
    }

    self.assertContainsSubset(want_components, run.components.keys())

    component = f'Transform.AutoData.{instance_name_2}'
    self.assertIs(
        type(
            run.components[component].outputs.transformed_examples.to_dataframe(
                'train', 5)), pd.DataFrame)


if __name__ == '__main__':
  e2etest.main()
