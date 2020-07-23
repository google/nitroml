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

from nitroml.components.meta_learning.meta_learner import executor
from nitroml.components.meta_learning.meta_feature_gen import MetaFeatures
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_component_specs
from tfx.types import standard_artifacts
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.types.component_spec import ChannelParameter
from tfx.types.artifact import Artifact


class MetaLearnerSpec(ComponentSpec):
  """MetaLearner component spec."""

  PARAMETERS = {
      'algorithm': ExecutionParameter(type=str),
  }
  INPUTS = {
      **{
          'meta_train_features_%s' % input_id:
          ChannelParameter(type=MetaFeatures, optional=True)
          for input_id in range(executor._MAX_INPUTS)
      }
  }
  OUTPUTS = {
      'metalearned_model': ChannelParameter(type=standard_artifacts.Model),
  }


class MetaLearner(base_component.BaseComponent):
  """MetaLearner that recommends a tuner config."""

  SPEC_CLASS = MetaLearnerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor.MetaLearnerExecutor)

  def __init__(self, algorithm: str, **meta_train_data: types.Channel = None):
    """Construct a NearestNeighborMetaLearner component.

    Args:
      meta_train_data: Dict of output of StatisticsGen for train datasets.
    """

    if not meta_train_stats:
      raise ValueError('Meta-train stats cannot be empty.')

    meta_features = types.Channel(type=MetaFeatures, artifacts=[MetaFeatures()])
    spec = MetaLearnerSpec(
        meta_features=meta_features, **meta_train_data)

    super(MetaLearner, self).__init__(spec=spec)
