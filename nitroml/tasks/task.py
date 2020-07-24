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
r"""An AutoML benchmark tasks."""

import abc
from typing import List
from tfx import types
from tfx.components.base import base_component

from nitroml.protos import problem_statement_pb2 as ps_pb2


class Task(abc.ABC):
  r"""Defines a Task for AutoML."""

  @abc.abstractproperty
  def name(self) -> str:
    """Returns the task's name."""

  @abc.abstractproperty
  def components(self) -> List[base_component.BaseComponent]:
    """Returns TFX components required for the task."""

  @abc.abstractproperty
  def train_and_eval_examples(self) -> types.Channel:
    """Returns train and eval labeled examples."""

  @abc.abstractproperty
  def problem_statement(self) -> ps_pb2.ProblemStatement:
    """Returns the ProblemStatement associated with this Task."""
