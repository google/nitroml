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
"""A Transform TFX component that support a more flexible preprocessing_fns."""

from typing import Any, Dict, Optional, Text, Union

from nitroml.components.meta_learning.meta_feature_gen import executor
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.types.component_spec import ComponentSpec


class MetaFeatureGenSpec(ComponentSpec):
  """MetaFeatureGen component spec."""

  PARAMETERS = {}
  INPUTS = {
      'meta_train_statistics':
          ChannelParameter(type=standard_artifacts.ExampleStatistics),
      'meta_test_statistics':
          ChannelParameter(type=standard_artifacts.ExampleStatistics)
  }
  OUTPUTS = {}


class MetaFeatureGen(base_component.BaseComponent):
  """MetaFeatureGen component that generates meta features."""

  SPEC_CLASS = MetaFeatureGenSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor.MetaFeatureGenExecutor)

  def __init__(self,
               meta_train_statistics: types.Channel = None,
               meta_test_statistics: types.Channel = None,
               instance_name: Optional[Text] = None):
    """Construct a MetaFeatureGen component.

    Args:
      meta_train_statistics: List of output of StatisticsGen for train datasets.
      meta_test_statistics: List of output of StatisticsGen for test datasets.
    """

    if not meta_train_statistics:
      raise ValueError('')
    spec = standard_component_specs.TransformSpec(
        meta_train_statistics=meta_train_statistics,
        meta_test_statistics=meta_test_statistics)

    super(MetaFeatureGen, self).__init__(spec=spec)
