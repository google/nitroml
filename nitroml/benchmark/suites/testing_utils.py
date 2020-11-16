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
"""Utilities for testing datasets."""

import json
import os
from typing import List

from nitroml.benchmark.suites import data_utils
from nitroml.benchmark.suites import openml_cc18
import requests_mock
import tensorflow as tf


def register_mock_urls(mocker: requests_mock.Mocker,
                       dataset_id_list: List[int] = None):
  """Registers mock URLs to the Mocker."""

  if not dataset_id_list:
    dataset_id_list = [1, 2]

  list_url = f'{openml_cc18._OPENML_API_URL}/data/list'  # pylint: disable=protected-access

  for name, value in data_utils.parse_dataset_filters(
      openml_cc18._DATASET_FILTERS).items():  # pylint: disable=protected-access
    list_url = f'{list_url}/{name}/{value}'

  with tf.io.gfile.GFile(
      _filename_path('openml_list.get.json'), mode='r') as fin:
    mocker.get(list_url, json=json.load(fin), status_code=200)

  for dataset_id in dataset_id_list:

    desc_url = f'{openml_cc18._OPENML_API_URL}/data/{dataset_id}'  # pylint: disable=protected-access

    with tf.io.gfile.GFile(
        _filename_path(f'openml_description.get_{dataset_id}.json'),
        mode='r') as fin:
      mocker.get(desc_url, json=json.load(fin), status_code=200)

    qual_url = f'{openml_cc18._OPENML_API_URL}/data/qualities/{dataset_id}'  # pylint: disable=protected-access

    with tf.io.gfile.GFile(
        _filename_path('openml_qual.get.json'), mode='r') as fin:
      mocker.get(qual_url, json=json.load(fin), status_code=200)

    csv_url = f'{openml_cc18._OPENML_FILE_API_URL}/get_csv/{dataset_id}'  # pylint: disable=protected-access

    with tf.io.gfile.GFile(_filename_path('dataset.txt'), mode='r') as fin:
      mocker.get(csv_url, text=fin.read())


def _filename_path(filename: str) -> str:
  return os.path.join(
      os.path.dirname(__file__), '../tasks/testdata/openml_cc18', filename)
