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
"""Tests for nitroml.results."""
import os

from absl.testing import absltest
from absl.testing import parameterized

from nitroml.benchmark import result as br
from nitroml.benchmark import results
from nitroml.testing import test_mlmd
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2

_MLMD_02_05_21_PATH = os.path.join(
    os.path.dirname(__file__), 'testdata/mlmd/mlmd_02_05_21.sqlite')


class OverviewTest(parameterized.TestCase):
  """Tests nitroml.results.overview."""

  @parameterized.named_parameters(
      {
          'testcase_name':
              'no aggregation 02-05-21',
          'mlmd_store_path':
              _MLMD_02_05_21_PATH,
          'metric_aggregators':
              None,
          'want_columns': [
              'run_id', 'benchmark_fullname', 'benchmark', 'run', 'num_runs',
              'accuracy', 'average_loss', 'post_export_metrics/example_count'
          ],
      }, {
          'testcase_name':
              'mean 02-05-21',
          'mlmd_store_path':
              _MLMD_02_05_21_PATH,
          'metric_aggregators': ['mean'],
          'want_columns': [
              'run_id', 'benchmark', 'num_runs', 'accuracy mean',
              'average_loss mean', 'post_export_metrics/example_count mean'
          ],
      }, {
          'testcase_name':
              'mean and stdev 02-05-21',
          'mlmd_store_path':
              _MLMD_02_05_21_PATH,
          'metric_aggregators': ['mean', 'std'],
          'want_columns': [
              'run_id', 'benchmark', 'num_runs', 'accuracy mean',
              'accuracy std', 'average_loss mean', 'average_loss std',
              'post_export_metrics/example_count mean',
              'post_export_metrics/example_count std'
          ],
      })
  def test_overview(self, mlmd_store_path, metric_aggregators, want_columns):
    config = metadata_store_pb2.ConnectionConfig()
    config.sqlite.filename_uri = mlmd_store_path

    store = metadata_store.MetadataStore(config)
    df = results.overview(store, metric_aggregators=metric_aggregators)
    self.assertEqual(want_columns, df.columns.tolist())


class ToPyTypeTest(absltest.TestCase):

  def testIntVal(self):
    val = results._to_pytype('12345')
    self.assertEqual(val, 12345)

  def testFloatVal(self):
    val = results._to_pytype('123.45')
    self.assertEqual(val, 123.45)

  def testBoolVal(self):
    true_val = results._to_pytype('True')
    false_val = results._to_pytype('False')
    self.assertEqual(true_val, True)
    self.assertEqual(false_val, False)

  def testStringVal(self):
    val = results._to_pytype('Awesome')
    self.assertEqual(val, 'Awesome')


class GetBenchmarkResultsTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(GetBenchmarkResultsTest, self).__init__(*args, **kwargs)
    self.test_mlmd = test_mlmd.TestMLMD()

  def testGetBenchmarkResults(self):
    run_id = 'run_id'
    artifact_id = self.test_mlmd.put_artifact({
        'accuracy': '0.25',
        'average_loss': '2.40',
        'name': 'pipeline:run_id:component:artifact:0',
        br.BenchmarkResult.BENCHMARK_NAME_KEY: 'Test'
    })
    execution_id = self.test_mlmd.put_execution(run_id)
    self.test_mlmd.put_event(artifact_id, execution_id)

    result = results._get_benchmark_results(self.test_mlmd.store)
    properties = result.properties['run_id.Test']
    property_names = result.property_names
    self.assertEqual(0.25, properties['accuracy'])
    self.assertEqual(2.4, properties['average_loss'])
    self.assertEqual('Test', properties[br.BenchmarkResult.BENCHMARK_NAME_KEY])
    self.assertEqual('run_id', properties[results.RUN_ID_KEY])
    self.assertEqual(['accuracy', 'average_loss'], property_names)

  def testGetBenchmarkResultsTwoArtifacts(self):
    run_id = 'run_id'
    execution_id0 = self.test_mlmd.put_execution(run_id)
    artifact_id0 = self.test_mlmd.put_artifact({
        'accuracy': '0.25',
        'average_loss': '2.40',
        'name': 'pipeline:run_id:component:artifact:0',
        br.BenchmarkResult.BENCHMARK_NAME_KEY: 'Test'
    })
    self.test_mlmd.put_event(artifact_id0, execution_id0)
    execution_id1 = self.test_mlmd.put_execution(run_id)
    artifact_id1 = self.test_mlmd.put_artifact({
        'my_custom_metric': '99',
        'name': 'pipeline:run_id:component:artifact:1',
        br.BenchmarkResult.BENCHMARK_NAME_KEY: 'Test'
    })
    self.test_mlmd.put_event(artifact_id1, execution_id1)

    result = results._get_benchmark_results(self.test_mlmd.store)
    properties = result.properties['run_id.Test']
    property_names = result.property_names
    self.assertEqual(0.25, properties['accuracy'])
    self.assertEqual(2.4, properties['average_loss'])
    self.assertEqual(99, properties['my_custom_metric'])
    self.assertEqual('Test', properties[br.BenchmarkResult.BENCHMARK_NAME_KEY])
    self.assertEqual('run_id', properties[results.RUN_ID_KEY])
    self.assertEqual(['accuracy', 'average_loss', 'my_custom_metric'],
                     property_names)


class GetStatisticsGenDirectoryTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(GetStatisticsGenDirectoryTest, self).__init__(*args, **kwargs)
    self.test_mlmd = test_mlmd.TestMLMD(artifact_type=results._STATS)

  def testGetStatistictsGenDir(self):
    run_id = '0'
    artifact_id = self.test_mlmd.put_artifact({'test_element': 'test_output'})
    execution_id = self.test_mlmd.put_execution(run_id)
    self.test_mlmd.put_event(artifact_id, execution_id)

    expected_dirs = ['test/path/']
    result_dirs = results.get_statisticsgen_dir_list(self.test_mlmd.store)
    self.assertEqual(result_dirs, expected_dirs)


if __name__ == '__main__':
  absltest.main()
