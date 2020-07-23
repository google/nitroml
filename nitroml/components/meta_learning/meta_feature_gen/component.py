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
"""Implements MetaFeatureGen component."""

from typing import Any, Dict, Optional, Text, Union

from nitroml.components.meta_learning.meta_feature_gen import executor
from nitroml.components.meta_learning import artifacts
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.types.component_spec import ChannelParameter


class MetaFeatureGenSpec(ComponentSpec):
  """MetaFeatureGenSpec component spec."""

  PARAMETERS = {
      'custom_config': ExecutionParameter(type=(str, str), optional=True),
  }
  INPUTS = {
      'statistics': ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }
  OUTPUTS = {
      'meta_features': ChannelParameter(type=artifacts.MetaFeatures),
  }


class MetaFeatureGen(base_component.BaseComponent):
  """Custom MetaFeatureGen that generated meta-features for the dataset."""

  SPEC_CLASS = MetaFeatureGenSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor.MetaFeatureGenExecutor)

  def __init__(self,
               statistics: types.Channel = None,
               custom_config: Optional[Dict[str, Any]] = None,
               instance_name: Optional[str] = None):
    """Construct a MetaFeatureGen component.

    Args:
      statistics: Output of StatisticsGen.
      custom_config: A dict which contains addtional parameters.
      instance_name: Optional unique instance name. Necessary iff multiple
        MetaFeatureGen components are declared in the same pipeline.
    """

    meta_features = types.Channel(
        type=artifacts.MetaFeatures, artifacts=[artifacts.MetaFeatures()])
    spec = MetaFeatureGenSpec(
        meta_features=meta_features,
        statistics=statistics,
        custom_config=custom_config)
    super(MetaFeatureGen, self).__init__(spec=spec, instance_name=instance_name)
