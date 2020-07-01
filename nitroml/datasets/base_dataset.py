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
from typing import List, Text, Dict

from tfx.components.base import base_component


class BaseDataset(abc.ABC):
  """Dataset abstract class"""

  def __init__(self, data_dir: Text = '/tmp/', max_threads: int = 1):
    """Base dataset class used as a superclass for downloading datasets not provided by tfds."""

    assert max_threads > 0

    self._components = None
    self._tasks = None
    self.root_dir = data_dir
    self.max_threads = max_threads
    super().__init__()

  @property
  def components(self) -> List[base_component.BaseComponent]:
    """Returns the components for tfx pipeline."""

    if self._components:
      return self._components

    return self._create_components()

  @property
  def tasks(self) -> List[Dict[Text, Text]]:
    """Returns the list of task information for the openML datasets"""

    if self._tasks:
      return self._tasks

    return self._create_tasks()

  @abc.abstractmethod
  def _create_components(self) -> List[base_component.BaseComponent]:
    """Create the components for the tfx pipeline."""

    pass

  @abc.abstractmethod
  def _create_tasks(self) -> List[Dict[Text, Text]]:
    """Loads the task information from disk."""

  pass