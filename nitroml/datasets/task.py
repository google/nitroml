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

import json

from typing import Text


class Task(object):
  r"""Task class to track the task associated with the benchmark."""

  # Note: We may have to clear the saved task objects if we change these constants.
  BINARY_CLASSIFICATION = 'binary_classification'
  CATEGORICAL_CLASSIFICATION = 'categorical_classification'
  REGRESSION = 'regression'

  def __init__(self,
               task_type: Text = None,
               num_classes: int = 0,
               label_key: Text = None,
               description: Text = ''):
    super().__init__()

    assert self._verify_task(task_type), 'Invalid task type'
    self._type = task_type
    self._num_classes = num_classes
    self._description = description
    self._label_key = label_key

  def __str__(self):
    return (
        f'Task: {self._type} \nClasses: {self._num_classes} \nDescription: {self._description}'
    )

  def _verify_task(self, task_type: Text = None):

    if task_type == Task.BINARY_CLASSIFICATION:
      return True
    elif task_type == Task.CATEGORICAL_CLASSIFICATION:
      return True
    elif task_type == Task.REGRESSION:
      return True

    return False

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
    return_dict = {}
    return_dict['description'] = self._description
    return_dict['type'] = self._type
    return_dict['num_classes'] = self._num_classes
    return_dict['label_key'] = self._label_key
    # return json.dumps(return_dict, sort_keys=True, indent=4)
    return return_dict