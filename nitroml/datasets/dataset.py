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
r"""Abstract Dataset class used for manually downloading datasets."""

import abc
import datetime
import functools
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Text

from absl import logging
from tfx.components.base import base_component
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.utils.dsl_utils import external_input

from nitroml.datasets import data_utils, task


class Dataset(abc.ABC):
  """Dataset abstract class"""

  def __init__(self, data_dir: Text = '/tmp/', max_threads: int = 1):

    assert max_threads > 0

    self._components = None
    self._tasks = None
    self.root_dir = data_dir
    self.max_threads = max_threads
    super().__init__()

  @abc.abstractmethod
  def _create_components(self) -> List[base_component.BaseComponent]:
    pass

  @property
  def components(self):
    if self._components:
      return self._components
    else:
      return self._create_components()

  @abc.abstractmethod
  def _create_tasks(self) -> List[task.Task]:
    pass

  @property
  def tasks(self):
    if self._tasks:
      return self._tasks
    else:
      return self._create_tasks()
