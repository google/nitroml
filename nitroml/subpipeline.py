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
"""A subpipeline definition."""

import abc
from typing import List

from tfx.components.base import base_component
from tfx.types import node_common

SubpipelineOutputs = node_common._PropertyDictWrapper  # pylint: disable=protected-access


class Subpipeline(abc.ABC):
  """A subpipeline abstract class for the NitroML framework.

  Subpipeline allows different subpipeline objects to inherit a consistent API.
  """

  @abc.abstractproperty
  def id(self) -> str:
    """Returns the string ID."""

  @abc.abstractproperty
  def components(self) -> List[base_component.BaseComponent]:
    """Returns the components of this subpipeline."""

  @abc.abstractproperty
  def outputs(self) -> SubpipelineOutputs:
    """Returns the Subpipeline's outputs."""
