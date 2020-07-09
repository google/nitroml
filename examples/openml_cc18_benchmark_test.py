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

import json
import os
import sys
import tempfile

from absl import flags
from absl.testing import absltest
import requests_mock
import tensorflow as tf
import nitroml
from nitroml import results
from examples import openml_cc18_benchmark
from nitroml.datasets import openml_cc18
from nitroml.datasets import data_utils
from nitroml.testing import e2etest

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS


class OpenMLCC18BenchmarkTest(e2etest.TestCase):

  def setUp(self):
    super(OpenMLCC18BenchmarkTest, self).setUp()
    flags.FLAGS(sys.argv)

    tempdir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self._pipeline_name = "nitroml_openml_cc_18_benchmark"
    self._pipeline_root = os.path.join(tempdir, "nitroml", "openml",
                                       self._pipeline_name)
    self._metadata_path = os.path.join(tempdir, "nitroml", "openml",
                                       self._pipeline_name, "metadata.db")
    self._data_dir = os.path.join(tempdir, 'openML_mock_data')

  def test(self):
    metadata_config = metadata_store_pb2.ConnectionConfig(
        sqlite=metadata_store_pb2.SqliteMetadataSourceConfig(
            filename_uri=self._metadata_path))

    with requests_mock.Mocker() as mocker:
      self.register_mock_urls(mocker)
      nitroml.run(
          [openml_cc18_benchmark.OpenMLCC18Benchmark()],
          pipeline_name=self._pipeline_name,
          pipeline_root=self._pipeline_root,
          metadata_connection_config=metadata_config,
          data_dir=self._data_dir,
          mock_data=True,
      )

    instance_name = '.'.join(['OpenMLCC18Benchmark', 'benchmark', 'mockdata'])
    self.assertComponentExecutionCount(7, self._metadata_path)
    self.assertComponentSucceeded('.'.join(["CsvExampleGen", instance_name]),
                                  self._metadata_path)
    self.assertComponentSucceeded('.'.join(["SchemaGen", instance_name]),
                                  self._metadata_path)
    self.assertComponentSucceeded('.'.join(["StatisticsGen", instance_name]),
                                  self._metadata_path)
    self.assertComponentSucceeded('.'.join(["Transform", instance_name]),
                                  self._metadata_path)
    self.assertComponentSucceeded('.'.join(["Trainer", instance_name]),
                                  self._metadata_path)
    self.assertComponentSucceeded('.'.join(["Evaluator", instance_name]),
                                  self._metadata_path)
    self.assertComponentSucceeded(
        '.'.join(["BenchmarkResultPublisher", instance_name]),
        self._metadata_path)

    # Load benchmark results.
    store = metadata_store.MetadataStore(metadata_config)
    df = results.overview(store)

    # Check benchmark results overview values.
    self.assertEqual(len(df.index), 1)
    self.assertContainsSubset([
        "benchmark",
        "run",
        "num_runs",
        "accuracy",
        "average_loss",
        "post_export_metrics/example_count",
    ], df.columns.values.tolist())
    self.assertSameElements([1], df["run"].tolist())
    self.assertSameElements([1], df["num_runs"].tolist())
    self.assertSameElements([instance_name], df.benchmark.unique())

  def register_mock_urls(self, mocker):

    list_url = f'{openml_cc18._OPENML_API_URL}/data/list'
    dataset_id = 1

    for name, value in data_utils.parse_dataset_filters(
        openml_cc18._DATASET_FILTERS).items():
      list_url = f'{list_url}/{name}/{value}'

    with tf.io.gfile.GFile(
        _filename_path('openml_list.get.json'), mode='r') as fin:
      mocker.get(list_url, json=json.load(fin), status_code=200)

    desc_url = f'{openml_cc18._OPENML_API_URL}/data/{dataset_id}'

    with tf.io.gfile.GFile(
        _filename_path('openml_description.get.json'), mode='r') as fin:
      mocker.get(desc_url, json=json.load(fin), status_code=200)

    qual_url = f'{openml_cc18._OPENML_API_URL}/data/qualities/{dataset_id}'

    with tf.io.gfile.GFile(
        _filename_path('openml_qual.get.json'), mode='r') as fin:
      mocker.get(qual_url, json=json.load(fin), status_code=200)

    csv_url = f'{openml_cc18._OPENML_FILE_API_URL}/get_csv/{dataset_id}'

    with tf.io.gfile.GFile(_filename_path('dataset.txt'), mode='r') as fin:
      mocker.get(csv_url, text=fin.read())


def _filename_path(filename: str) -> str:
  return os.path.join(
      os.path.dirname(__file__), '..', 'nitroml', 'datasets', 'testdata',
      filename)


if __name__ == "__main__":
  absltest.main()
