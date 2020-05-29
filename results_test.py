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

import datetime
import os
from typing import Dict

from absl.testing import absltest
from absl.testing import parameterized

from nitroml import results
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

_MLMD_03_31_20_PATH = os.path.join(
    os.path.dirname(__file__), 'testdata/mlmd/mlmd_03_31_20.sqlite')
_MLMD_04_01_20_PATH = os.path.join(
    os.path.dirname(__file__), 'testdata/mlmd/mlmd_04_01_20.sqlite')
_MLMD_05_21_20_PATH = os.path.join(
    os.path.dirname(__file__), 'testdata/mlmd/mlmd_05_21_20.sqlite')


class OverviewTest(parameterized.TestCase):
  """Tests nitroml.results.overview."""

  @parameterized.named_parameters(
      {
          'testcase_name':
              'no aggregation 03-31-20',
          'mlmd_store_path':
              _MLMD_03_31_20_PATH,
          'metric_aggregators':
              None,
          'want_columns': [
              'run_id',
              'benchmark_fullname',
              'benchmark',
              'eval_batch_size',
              'layer_size',
              'num_layers',
              'train_batch_size',
              'accuracy',
              'accuracy_baseline',
              'auc',
              'auc_precision_recall',
              'average_loss',
              'label/mean',
              'pipeline_name',
              'post_export_metrics/example_count',
              'precision',
              'prediction/mean',
              'recall',
          ],
      }, {
          'testcase_name':
              'mean 03-31-20',
          'mlmd_store_path':
              _MLMD_03_31_20_PATH,
          'metric_aggregators': ['mean'],
          'want_columns': [
              'run_id',
              'benchmark',
              'eval_batch_size',
              'layer_size',
              'num_layers',
              'train_batch_size',
              'accuracy mean',
              'accuracy_baseline mean',
              'auc mean',
              'auc_precision_recall mean',
              'average_loss mean',
              'label/mean mean',
              'post_export_metrics/example_count mean',
              'precision mean',
              'prediction/mean mean',
              'recall mean',
          ],
      }, {
          'testcase_name':
              'mean and stdev 03-31-20',
          'mlmd_store_path':
              _MLMD_03_31_20_PATH,
          'metric_aggregators': ['mean', 'std'],
          'want_columns': [
              'run_id',
              'benchmark',
              'eval_batch_size',
              'layer_size',
              'num_layers',
              'train_batch_size',
              'accuracy mean',
              'accuracy std',
              'accuracy_baseline mean',
              'accuracy_baseline std',
              'auc mean',
              'auc std',
              'auc_precision_recall mean',
              'auc_precision_recall std',
              'average_loss mean',
              'average_loss std',
              'label/mean mean',
              'label/mean std',
              'post_export_metrics/example_count mean',
              'post_export_metrics/example_count std',
              'precision mean',
              'precision std',
              'prediction/mean mean',
              'prediction/mean std',
              'recall mean',
              'recall std',
          ],
      }, {
          'testcase_name':
              'no aggregation 04-01-20',
          'mlmd_store_path':
              _MLMD_04_01_20_PATH,
          'metric_aggregators':
              None,
          'want_columns': [
              'run_id',
              'benchmark_fullname',
              'benchmark',
              'run',
              'num_runs',
              'eval_batch_size',
              'layer_size',
              'num_layers',
              'train_batch_size',
              'accuracy',
              'accuracy_baseline',
              'auc',
              'auc_precision_recall',
              'average_loss',
              'label/mean',
              'pipeline_name',
              'post_export_metrics/example_count',
              'precision',
              'prediction/mean',
              'recall',
          ],
      }, {
          'testcase_name':
              'mean 04-01-20',
          'mlmd_store_path':
              _MLMD_04_01_20_PATH,
          'metric_aggregators': ['mean'],
          'want_columns': [
              'run_id',
              'benchmark',
              'num_runs',
              'eval_batch_size',
              'layer_size',
              'num_layers',
              'train_batch_size',
              'accuracy mean',
              'accuracy_baseline mean',
              'auc mean',
              'auc_precision_recall mean',
              'average_loss mean',
              'label/mean mean',
              'post_export_metrics/example_count mean',
              'precision mean',
              'prediction/mean mean',
              'recall mean',
          ],
      }, {
          'testcase_name':
              'mean and stdev 04-01-20',
          'mlmd_store_path':
              _MLMD_04_01_20_PATH,
          'metric_aggregators': ['mean', 'std'],
          'want_columns': [
              'run_id',
              'benchmark',
              'num_runs',
              'eval_batch_size',
              'layer_size',
              'num_layers',
              'train_batch_size',
              'accuracy mean',
              'accuracy std',
              'accuracy_baseline mean',
              'accuracy_baseline std',
              'auc mean',
              'auc std',
              'auc_precision_recall mean',
              'auc_precision_recall std',
              'average_loss mean',
              'average_loss std',
              'label/mean mean',
              'label/mean std',
              'post_export_metrics/example_count mean',
              'post_export_metrics/example_count std',
              'precision mean',
              'precision std',
              'prediction/mean mean',
              'prediction/mean std',
              'recall mean',
              'recall std',
          ],
      }, {
          'testcase_name':
              'no aggregation 05-21-20',
          'mlmd_store_path':
              _MLMD_05_21_20_PATH,
          'metric_aggregators': None,
          'want_columns': [
              'run_id',
              'benchmark_fullname',
              'benchmark',
              'run',
              'num_runs',
              'eval_batch_size',
              'layer_size',
              'num_layers',
              'train_batch_size',
              'accuracy',
              'accuracy_baseline',
              'auc',
              'auc_precision_recall',
              'average_loss',
              'label/mean',
              'pipeline_name',
              'post_export_metrics/example_count',
              'precision',
              'prediction/mean',
              'recall',
              'kaggle_date',
              'kaggle_description',
              'kaggle_errorDescription',
              'kaggle_fileName',
              'kaggle_privateScore',
              'kaggle_publicScore',
              'kaggle_ref',
              'kaggle_status',
              'kaggle_submittedBy',
              'kaggle_submittedByRef',
              'kaggle_teamName',
              'kaggle_totalBytes',
              'kaggle_type',
              'kaggle_url'
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


class ParseHparamsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'empty_list',
          'hp_prop': '[]',
          'want': dict()
      }, {
          'testcase_name':
              'number_val',
          'hp_prop':
              "['batch_size=256', 'learning_rate=0.05', 'decay_rate=0.95']",
          'want': {
              'batch_size': 256,
              'learning_rate': 0.05,
              'decay_rate': 0.95
          }
      }, {
          'testcase_name':
              'bool_val',
          'hp_prop':
              "['has_batch_size=true', 'has_learning_rate=False', 'has_decay_rate=false']",
          'want': {
              'has_batch_size': True,
              'has_learning_rate': False,
              'has_decay_rate': False
          }
      }, {
          'testcase_name':
              'mix_vals',
          'hp_prop':
              "['has_batch_size=true','batch_size=256', 'learning_rate=0.05', 'is_awesome=yes']",
          'want': {
              'has_batch_size': True,
              'batch_size': 256,
              'learning_rate': 0.05,
              'is_awesome': 'yes'
          }
      })
  def testParseHparams(self, hp_prop, want):
    hparams = results._parse_hparams(hp_prop)
    self.assertEqual(hparams, want)


class MergeResultTest(absltest.TestCase):

  def testEmptyMergeResults(self):
    result1 = results._Result(properties={}, property_names=[])
    result2 = results._Result(properties={}, property_names=[])

    merge_result = results._merge_results([result1, result2])

    self.assertEqual(
        results._Result(properties={}, property_names=[]), merge_result)

  def testOneEmptyMergeResults(self):
    result1 = results._Result(
        properties={'key': {
            'nkey': 'val'
        }}, property_names=['test'])
    result2 = results._Result(properties={}, property_names=[])

    merge_result = results._merge_results([result1, result2])

    self.assertEqual(
        results._Result(
            properties={'key': {
                'nkey': 'val'
            }}, property_names=['test']), merge_result)

  def testMergeResults(self):
    result1 = results._Result(
        properties={
            'key1': {
                'hparam': 'val'
            },
            'key2': {
                'hparam': 'val'
            }
        },
        property_names=['hparam_names'])
    result2 = results._Result(
        {
            'key1': {
                'metrics': 'val'
            },
            'key3': {
                'metrics': 'val'
            }
        },
        property_names=['metric_names'])

    merge_result = results._merge_results([result1, result2])

    want_result = results._Result(
        properties={
            'key1': {
                'hparam': 'val',
                'metrics': 'val'
            },
            'key2': {
                'hparam': 'val'
            },
            'key3': {
                'metrics': 'val'
            }
        },
        property_names=['hparam_names', 'metric_names'])
    self.assertEqual(want_result, merge_result)


class GetHparamsTest(absltest.TestCase):

  def setUp(self):
    super(GetHparamsTest, self).setUp()
    config = metadata_store_pb2.ConnectionConfig()
    config.fake_database.SetInParent()
    self.store = metadata_store.MetadataStore(config)
    self.exec_type_id = self._put_execution_type()

  def _put_execution_type(self) -> int:
    exec_type = metadata_store_pb2.ExecutionType()
    exec_type.name = results._TRAINER
    exec_type.properties[results.RUN_ID_KEY] = metadata_store_pb2.STRING
    exec_type.properties[results._HPARAMS] = metadata_store_pb2.STRING
    exec_type.properties[results._COMPONENT_ID] = metadata_store_pb2.STRING
    return self.store.put_execution_type(exec_type)

  def _put_execution(self, run_id: str, trainer_name: str, hparam: str) -> int:
    execution = metadata_store_pb2.Execution()
    execution.properties[results._HPARAMS].string_value = hparam
    execution.properties[results.RUN_ID_KEY].string_value = run_id
    execution.properties[results._COMPONENT_ID].string_value = trainer_name
    execution.type_id = self.exec_type_id
    self.store.put_executions([execution])

  def testGetHparams(self):
    hparam = "['batch_size=256', 'learning_rate=0.05', 'decay_rate=0.95']"
    run_id = '0'
    trainer_name = results._TRAINER_PREFIX + '.Test'
    self._put_execution(run_id, trainer_name, hparam)

    result = results._get_hparams(self.store)

    want_result = results._Result(
        properties={
            '0.Test': {
                'batch_size': 256,
                'learning_rate': 0.05,
                'decay_rate': 0.95,
                results.RUN_ID_KEY: '0',
                results.BENCHMARK_KEY: 'Test',
                results.STARTED_AT: datetime.datetime.fromtimestamp(0)
            }
        },
        property_names=['batch_size', 'decay_rate', 'learning_rate'])
    self.assertEqual(want_result, result)


class GetBenchmarkResultsTest(absltest.TestCase):

  def setUp(self):
    super(GetBenchmarkResultsTest, self).setUp()
    config = metadata_store_pb2.ConnectionConfig()
    config.fake_database.SetInParent()
    self.store = metadata_store.MetadataStore(config)
    self.exec_type_id = self._put_execution_type()
    self.artifact_type_id = self._put_artifact_type()

  def _put_execution_type(self) -> int:
    exec_type = metadata_store_pb2.ExecutionType()
    exec_type.name = 'BenchmarkResultPublisher'
    exec_type.properties[results.RUN_ID_KEY] = metadata_store_pb2.STRING
    return self.store.put_execution_type(exec_type)

  def _put_artifact_type(self) -> int:
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = results._BENCHMARK_RESULT
    return self.store.put_artifact_type(artifact_type)

  def _put_execution(self, run_id: str) -> int:
    execution = metadata_store_pb2.Execution()
    execution.properties[results.RUN_ID_KEY].string_value = run_id
    execution.type_id = self.exec_type_id
    return self.store.put_executions([execution])[0]

  def _put_artifact(self, properties: Dict[str, str]) -> int:
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = self.artifact_type_id
    for name, val in properties.items():
      artifact.custom_properties[name].string_value = val
    return self.store.put_artifacts([artifact])[0]

  def _put_event(self, artifact_id: int, execution_id: int) -> None:
    event = metadata_store_pb2.Event()
    event.type = metadata_store_pb2.Event.OUTPUT
    event.artifact_id = artifact_id
    event.execution_id = execution_id
    self.store.put_events([event])

  def testGetBenchmarkResults(self):
    run_id = '0'
    artifact_id = self._put_artifact({
        'accuracy': '0.25',
        'average_loss': '2.40',
        results.BENCHMARK_KEY: 'Test'
    })
    execution_id = self._put_execution(run_id)
    self._put_event(artifact_id, execution_id)

    result = results._get_benchmark_results(self.store)

    want_result = results._Result(
        properties={
            '0.Test': {
                'accuracy': 0.25,
                'average_loss': 2.40,
                results.RUN_ID_KEY: '0',
                results.BENCHMARK_KEY: 'Test',
                results.STARTED_AT: datetime.datetime.fromtimestamp(0)
            }
        },
        property_names=['accuracy', 'average_loss'])
    self.assertEqual(want_result, result)


if __name__ == '__main__':
  absltest.main()
