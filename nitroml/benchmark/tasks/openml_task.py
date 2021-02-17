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

import os
from typing import List

import nitroml
from tfx import components as tfx
from tfx import types
from tfx.dsl.components.base import base_component

from nitroml.protos import problem_statement_pb2 as ps_pb2


class OpenMLTask(nitroml.BenchmarkTask):
  r"""Defines an OpenML task for an AutoML pipeline."""

  # Note: We may have to clear the saved task objects if we change these
  # constants.
  BINARY_CLASSIFICATION = 'binary_classification'
  CATEGORICAL_CLASSIFICATION = 'categorical_classification'
  REGRESSION = 'regression'

  def __init__(self,
               name: str,
               root_dir: str,
               dataset_name: str,
               task_type: str,
               label_key: str,
               num_classes: int = 0,
               description: str = ''):
    if not self._verify_task(task_type):
      raise ValueError('Invalid task type')

    self._name = name
    self._dataset_name = dataset_name
    self._type = task_type
    self._num_classes = num_classes
    self._description = description
    self._label_key = label_key
    # TODO(nikhilmehta, weill): Subbenchmarking also appends task.name
    # to the component_id. Fix this when variable scoping is introduced.
    self._example_gen = tfx.CsvExampleGen(
        input_base=os.path.join(root_dir, f'{dataset_name}', 'data'),
        instance_name=self.name)

  def __str__(self):
    return (f'Task: {self._type} \nClasses: {self._num_classes} \n'
            f'Description: {self._description}')

  def _verify_task(self, task_type: str = None) -> bool:
    """Verify if task_type is among valid tasks or not."""

    return task_type in [
        self.BINARY_CLASSIFICATION, self.CATEGORICAL_CLASSIFICATION,
        self.REGRESSION
    ]

  @property
  def name(self) -> str:
    return f'OpenML.{self._name}'

  @property
  def dataset_name(self) -> str:
    return self._dataset_name

  @property
  def type(self) -> str:
    return self._type

  @property
  def num_classes(self) -> int:
    return self._num_classes

  @property
  def label_key(self) -> str:
    return self._label_key

  @property
  def components(self) -> List[base_component.BaseComponent]:
    return [self._example_gen]

  @property
  def train_and_eval_examples(self) -> types.Channel:
    """Returns train and eval labeled examples."""

    return self._example_gen.outputs.examples

  @property
  def test_examples(self) -> types.Channel:
    """Returns test labeled examples."""

    return self._example_gen.outputs.examples

  @property
  def test_example_splits(self) -> List[str]:
    return ['eval']

  @property
  def problem_statement(self) -> ps_pb2.ProblemStatement:
    """Returns the ProblemStatement associated with this Task."""

    return ps_pb2.ProblemStatement(
        owner=['nitroml'],
        tasks=[ps_pb2.Task(
            name=self.name,
            type=self._get_task_type(),
        )])

  def _get_task_type(self):
    """Creates a `ps_pb2.Type` from the number of classes."""

    if self.num_classes == 0:
      return ps_pb2.Type(
          one_dimensional_regression=ps_pb2.OneDimensionalRegression(
              label=self._label_key))
    if self.num_classes == 2:
      return ps_pb2.Type(
          binary_classification=ps_pb2.BinaryClassification(
              label=self._label_key))
    return ps_pb2.Type(
        multi_class_classification=ps_pb2.MultiClassClassification(
            label=self._label_key))
