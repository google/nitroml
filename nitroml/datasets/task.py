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
r"""Task class to identify the type of problem."""


class Task:
  r"""Task class to track the task associated with the benchmark."""

  # Note: We may have to clear the saved task objects if we change these
  # constants.
  BINARY_CLASSIFICATION = 'binary_classification'
  CATEGORICAL_CLASSIFICATION = 'categorical_classification'
  REGRESSION = 'regression'

  def __init__(self,
               dataset_name: str,
               task_type: str,
               label_key: str,
               num_classes: int = 0,
               description: str = ''):
    super().__init__()

    if not self._verify_task(task_type):
      raise ValueError('Invalid task type')

    self._dataset_name = dataset_name
    self._type = task_type
    self._num_classes = num_classes
    self._description = description
    self._label_key = label_key

  def __str__(self):
    return (f'Task: {self._type} \nClasses: {self._num_classes} \n'
            f'Description: {self._description}')

  def _verify_task(self, task_type: str = None) -> bool:
    """Verify if task_type is among valid tasks or not."""

    return task_type in [
        Task.BINARY_CLASSIFICATION, Task.CATEGORICAL_CLASSIFICATION,
        Task.REGRESSION
    ]

  @property
  def dataset_name(self):
    return self._dataset_name

  @property
  def type(self):
    return self._type

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def label_key(self):
    return self._label_key

  def to_dict(self):
    """Convert task attributes to dictionary."""

    return_dict = {}
    return_dict['dataset_name'] = self._dataset_name
    return_dict['description'] = self._description
    return_dict['task_type'] = self._type
    return_dict['num_classes'] = self._num_classes
    return_dict['label_key'] = self._label_key
    return return_dict
