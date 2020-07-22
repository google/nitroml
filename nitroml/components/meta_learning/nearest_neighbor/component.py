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
"""Implements NearestNeighbor component."""

from typing import Any, Dict, Optional, Text, Union

from nitroml.components.meta_learning.nearest_neighbor import executor
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_component_specs
from tfx.types import standard_artifacts
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.types.component_spec import ChannelParameter
from tfx.types.artifact import Artifact


class MetaFeatures(Artifact):
  """NitroML's custom Artifact for meta features."""
  TYPE_NAME = 'NitroML.MetaFeatures'


class NearestNeighborMetaLearnerSpec(ComponentSpec):
  """NearestNeighborMetaLearnerSpec component spec."""

  PARAMETERS = {
      **{
          'train_tasks_%s' % input_id: ChannelParameter(
              type=standard_artifacts.ExampleStatistics, optional=True)
          for input_id in range(executor._MAX_INPUTS)
      }
  }
  INPUTS = {
      **{
          'train_statistics_%s' % input_id: ChannelParameter(
              type=standard_artifacts.ExampleStatistics, optional=True)
          for input_id in range(executor._MAX_INPUTS)
      }
  }
  OUTPUTS = {'meta_features': ChannelParameter(type=MetaFeatures)}


class NearestNeighborMetaLearner(base_component.BaseComponent):
  """NearestNeighborMetaLearner that recommends a tuner config."""

  SPEC_CLASS = NearestNeighborMetaLearnerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor.NearestNeighborMetaLearnerExecutor)

  def __init__(self, **meta_train_data):
    """Construct a NearestNeighborMetaLearner component.

    Args:
      meta_train_data: Dict of output of StatisticsGen for train datasets.
    """

    if not meta_train_stats:
      raise ValueError('Meta-train stats cannot be empty.')

    meta_features = types.Channel(type=MetaFeatures, artifacts=[MetaFeatures()])

    spec = NearestNeighborMetaLearnerSpec(
        meta_features=meta_features, **meta_train_data)

    super(NearestNeighborMetaLearner, self).__init__(spec=spec)
