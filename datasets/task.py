from enum import Enum
from typing import Text
import json


class Task(object):

  # Note: We may have to clear the saved task objects if we change these constants.
  BINARY_CLASSIFICATION = 'binary_classification'
  CATEGORICAL_CLASSIFICATION = 'categorical_classification'
  REGRESSION = 'regression'

  def __init__(self,
               task_type: Text = None,
               head: int = 0,
               label_key: Text = None,
               description: Text = ''):
    super().__init__()

    assert self._verify_task(task_type), 'Invalid task type'
    self._type = task_type
    self._head = head
    self._description = description
    self._label_key = label_key

  def __str__(self):
    return (
        f'Task: {self._type} \nClasses: {self._head} \nDescription: {self._description}'
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
  def head(self):
    return self._head

  def toJSON(self):
    return_dict = {}
    return_dict['type'] = self._type
    return_dict['head'] = self._head
    return_dict['label_key'] = self._label_key
    # return json.dumps(return_dict, sort_keys=True, indent=4)
    return return_dict
