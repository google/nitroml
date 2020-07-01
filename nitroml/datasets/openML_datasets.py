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
r"""The OpenML dataset provider."""

import datetime
import functools
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Text, Dict, Any

from absl import logging

from tfx.components.base import base_component
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.utils.dsl_utils import external_input
from nitroml.datasets import data_utils, task, base_dataset


class OpenMLDataset(base_dataset.BaseDataset):
  """The OpenMLDataset class which downloads the latest datasets provided by OpenML."""

  _OPENML_API_URL = 'https://www.openml.org/api/v1/json'
  _OPENML_FILE_API_URL = 'https://www.openml.org/data/v1'
  _DATASET_FILTERS = ['status=active', 'tag=OpenML-CC18']
  _API_KEY = 'b1514bb2761ecc4709ab26db50673a41'

  def __init__(self,
               root_dir: Text = None,
               use_cache: bool = True,
               max_threads: int = 1):
    """OpenMLDataset Handler """

    data_dir = os.path.join(root_dir, 'openML_datasets')
    self._names = None
    super().__init__(data_dir, max_threads)

    if use_cache and os.path.exists(data_dir):
      logging.info('The directory %s exists. %d datasets found', data_dir,
                   len(os.listdir(data_dir)))
    else:
      self._get_data()

  @property
  def names(self) -> List[Text]:
    """Return name of all OpenML datasets"""

    if self._names:
      return self._names
    else:
      return os.listdir(self.root_dir)

  def _load_task(self, dataset_name='car') -> Dict[Text, Text]:
    """Loads the task information for the argument dataset."""

    with open(os.path.join(self.root_dir, dataset_name, 'task/task.json'),
              'r') as fin:
      data = json.load(fin)

    return data

  def _create_components(self) -> List[base_component.BaseComponent]:
    """Creates and returns the list of components for OpenML datasets."""

    components = []
    names = []
    for dataset_name in os.listdir(self.root_dir):
      dataset_dir = os.path.join(self.root_dir, f'{dataset_name}/data')
      examples = external_input(dataset_dir)
      example_gen = CsvExampleGen(input=examples, instance_name=dataset_name)
      components.append(example_gen)
      names.append(dataset_name)

    self._names = names
    self._components = components
    return components

  def _create_tasks(self) -> List[Dict[Text, Text]]:
    """Creates and returns the list of task properties for openML datasets."""

    tasks = []
    for dataset_name in os.listdir(self.root_dir):
      tasks.append(self._load_task(dataset_name))

    self._tasks = tasks
    return tasks

  def _get_data(self):
    """Downloads openML datasets using the OpenML API."""

    assert self.root_dir, 'Output_root_dir cannot be empty'

    if not os.path.isdir(self.root_dir):
      os.mkdir(self.root_dir)

    datasets = self._list_datasets(
        data_utils.parse_dataset_filters(self._DATASET_FILTERS))
    logging.info(f'There are {len(datasets)} datasets.')
    datasets = self._latest_version_only(datasets)

    parallel_fns = [
        functools.partial(self._dump_dataset, dataset, self.root_dir)
        for dataset in datasets
    ]

    logging.info(
        f'Downloading {len(datasets)} datasets. This may take a while.')

    succeeded, skipped = 0, 0
    failed = {}

    with ThreadPoolExecutor(max_workers=self.max_threads) as pool:
      tasks = {
          pool.submit(fn): datasets[ix] for ix, fn in enumerate(parallel_fns)
      }

      for future in as_completed(tasks):

        exec_info = future.exception()

        if exec_info is None:

          success = future.result()
          if success:
            succeeded += 1
          else:
            skipped += 1

          logging.info(
              f'Succeeded={succeeded}, failed={len(failed)}, skipped={skipped}')

        else:
          failed[tasks[future]['did']] = exec_info
          logging.warn('Exception: ', exec_info)
          logging.info(
              f'Succeeded={succeeded}, failed={len(failed)}, skipped={skipped}')

    for dataset in failed:
      logging.warn(dataset)
      logging.warn(failed[dataset])
      logging.info('\n**********')

    logging.info(
        f'Done! Succeeded={succeeded}, failed={len(failed)}, skipped={skipped}')

  def _list_datasets(self, filters: Dict[Text, Text]) -> List[Text]:
    """Returns the list of names of all `active` datasets from the OpenML repository matching the `filters`.

    Args:
      filters: Parsed filters for the OpenML API.
    """

    url = f'{self._OPENML_API_URL}/data/list'
    for name, value in filters.items():
      url = f'{url}/{name}/{value}'

    url = f'{url}?api_key={self._API_KEY}'
    resp = data_utils.get(url).json()
    return resp['data']['dataset']

  def _latest_version_only(self, datasets) -> List[Text]:
    """Filters the datasets to only keep the latest versions.
    Args:
      datasets: Array of dataset objects
    Returns:
      list of filtered datasets
    """

    filtered = {}
    for dataset in datasets:
      name = dataset['name']
      version = dataset['version']
      if name not in filtered or version > filtered[name]['version']:
        filtered[name] = dataset

    return list(filtered.values())

  def _dump_dataset(self, dataset, root_dir):
    """Dumps the `dataset` to root_dir.
    The `dataset` is downloaded from OpenML as CSV, converted to tf.Example, and
    written. A `task` object is created for the dataset and written to the same directory.
    Args:
      dataset: The OpenML dataset to dump to root_dir (as returned by `list_datasets`).
      root_dir: The root dir where to dump all dataset artifacts.
    Returns:
      Whether the dataset was dumped.
    """

    dataset_id = dataset['did']

    # Get dataset file_id and target_name.
    description = self._get_dataset_description(dataset_id)

    file_id = description['file_id']
    target_name = description['default_target_attribute']

    if ',' in target_name:
      logging.info(f'Skipping multi-label Dataset(id={dataset_id}).')
      return False

    # Get the dataset n_classes.
    qualities = self._get_data_qualities(dataset_id)

    for quality in qualities:
      if quality['name'] == 'NumberOfClasses':
        value = quality['value']
        n_classes = int(float(value)) if value else 0
    if n_classes == 1:
      logging.warning(f'Skipping single-class Dataset(id={dataset_id}).')
      return False

    dataset_name = dataset['name']
    dataset_dir = os.path.join(root_dir, f'{dataset_name}')
    column_rename_dict = self._download_dataset(file_id, dataset_dir)
    target_name = column_rename_dict[f'"{target_name}"']

    task_type = self._get_task_type(n_classes)

    description = (
        f'OpenML dataset with id={dataset_id}, name={dataset_name}, '
        f'type={task_type}. Fetched from OpenML servers '
        f'on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

    task_desc = task.Task(
        task_type=task_type,
        num_classes=n_classes,
        label_key=target_name,
        description=description)

    task_dir = os.path.join(dataset_dir, 'task')
    if not os.path.isdir(task_dir):
      os.makedirs(task_dir)

    task_path = os.path.join(task_dir, 'task.json')
    with open(task_path, 'w') as fout:
      json.dump(task_desc.to_dict(), fout)

    logging.info(f'OpenML dataset with id={dataset_id}, name={dataset_name}, '
                 f'on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

    return True

  def get_csv(self, file_id) -> Text:
    """Downloads the  OpenML dataset corresponding to the file with `file_id`.
    N.B.: The OpenML `file_id` does not correspond to the OpenML `dataset_id`.
    Args:
      file_id: The id of the file to download from OpenML.
    Returns:
    The downloaded CSV.
    """

    resp = data_utils.get(f'{self._OPENML_FILE_API_URL}/get_csv/{file_id}')
    resp = resp.text.replace(', ', ',').replace(' ,', ',')
    return resp

  def _download_dataset(self, file_id, dataset_dir) -> Dict[Text, Any]:
    """Downloads the OpenML dataset in boqth CSV and tf.Example sformat.
    The columns are renamed to be valid python identifiers.
    Args:
      file_id: The OpenML file_id of the dataset to download.
      dataset_dir: The directory where to write the downloaded dataset.
    Returns:
      A dictionary of <original column name> -> <renamed column name>.
    """

    csv = self.get_csv(file_id)

    # Rename the columns in the CSV to be valid python identifiers. This ensures
    # the column names (and label in the problem_statement proto) are the same for
    # both the CSV and the tf.Example datasets.
    csv = csv.split('\n')
    columns = csv[0].split(',')
    column_rename_dict = data_utils.rename_columns(columns)
    csv[0] = ','.join([column_rename_dict[column] for column in columns])
    csv = '\n'.join(csv)

    dataset_dir = os.path.join(dataset_dir, 'data')
    if not os.path.isdir(dataset_dir):
      os.makedirs(dataset_dir)

    csv_path = os.path.join(dataset_dir, 'dataset.csv')
    with open(csv_path, 'w') as fout:
      fout.write(csv)
      return column_rename_dict

  def _get_data_qualities(self, dataset_id) -> Dict[Text, Any]:
    """Returns the qualities of the dataset as specified using the OpenML API
    with `dataset_id`.
    """

    url = f'{self._OPENML_API_URL}/data/qualities/{dataset_id}?api_key={self._API_KEY}'
    resp = data_utils.get(url).json()
    return resp['data_qualities']['quality']

  def _get_dataset_description(self, dataset_id) -> Text:
    """Returns the dataset description using the OpenML API.
    Args:
      `dataset_id`: The dataset id as required by the API.
    """

    resp = data_utils.get(
        f'{self._OPENML_API_URL}/data/{dataset_id}?api_key={self._API_KEY}'
    ).json()
    return resp['data_set_description']

  def _get_task_type(self, n_classes) -> Text:
    """ Get the task information from num_classes"""

    if n_classes == 2:
      return task.Task.BINARY_CLASSIFICATION
    elif n_classes > 2:
      return task.Task.CATEGORICAL_CLASSIFICATION

    return task.Task.REGRESSION
