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
"""Tests for nitroml.datasets.openML_datasets.py."""

import tempfile
import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import requests_mock

from nitroml.datasets import openml_datasets
from nitroml.datasets import data_utils

USE_MOCK_DATA = True


class OpenMLDatasetsTest(parameterized.TestCase, absltest.TestCase):
  """Test cases for datasets.openML_datasets provider.

  There are two test cases:
  1) Test Case 1: Downloads mock data for openML CC18 datasets.
  2) Test Case 2: Checks the cache, if data exists loads from the disk, else downloads all the datasets.
  """

  @parameterized.named_parameters(
      {
          'testcase_name': 'openML_default',
          'root_dir': os.path.join(tempfile.gettempdir(), 'openML_mock_data'),
          'use_cache': False,
          'mock_data': USE_MOCK_DATA
      }, {
          'testcase_name': 'openML_use-cache',
          'root_dir': os.path.join(tempfile.gettempdir(), 'openML_mock_data'),
          'use_cache': True,
          'mock_data': USE_MOCK_DATA
      })
  def test_examples(self, **test_args):

    mock_data = test_args.get('mock_data', True)

    if mock_data:
      with requests_mock.Mocker() as mocker:
        self.register_mock_urls(mocker)
        datasets = openml_datasets.OpenMLCC18(**test_args)
    else:
      datasets = openml_datasets.OpenMLCC18(**test_args)

    self.assertNotEqual(len(datasets.components), 0)
    self.assertEqual(len(datasets.tasks), len(datasets.components))

  def register_mock_urls(self, mocker):

    list_url = f'{openml_datasets._OPENML_API_URL}/data/list'
    dataset_id = 1

    for name, value in data_utils.parse_dataset_filters(
        openml_datasets._DATASET_FILTERS).items():
      list_url = f'{list_url}/{name}/{value}'

    with open(
        'nitroml/datasets/mock_data/openml_list.get.json',
        'r',
        encoding='utf-8') as fin:
      mocker.get(list_url, json=json.load(fin), status_code=200)

    desc_url = f'{openml_datasets._OPENML_API_URL}/data/{dataset_id}'
    with open(
        'nitroml/datasets/mock_data/openml_description.get.json',
        'r',
        encoding='utf-8') as fin:
      mocker.get(desc_url, json=json.load(fin), status_code=200)

    qual_url = f'{openml_datasets._OPENML_API_URL}/data/qualities/{dataset_id}'
    with open(
        'nitroml/datasets/mock_data/openml_qual.get.json',
        'r',
        encoding='utf-8') as fin:

      mocker.get(qual_url, json=json.load(fin), status_code=200)

    csv_url = f'{openml_datasets._OPENML_FILE_API_URL}/get_csv/{dataset_id}'
    with open(
        'nitroml/datasets/mock_data/dataset.txt', 'r', encoding='utf-8') as fin:
      mocker.get(csv_url, text=fin.read())


if __name__ == '__main__':
  absltest.main()
