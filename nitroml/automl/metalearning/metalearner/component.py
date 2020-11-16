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
"""Implements the metalearner component."""

from typing import Any, Dict, Optional

from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.metalearner import executor
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class MetaLearnerSpec(ComponentSpec):
  """MetaLearner component spec."""

  PARAMETERS = {
      'algorithm': ExecutionParameter(type=str),
      'custom_config': ExecutionParameter(type=Dict[str, Any], optional=True),
  }
  INPUTS = {
      **{
          'meta_train_features_%s' % input_id:
          ChannelParameter(type=artifacts.MetaFeatures, optional=True)
          for input_id in range(executor.MAX_INPUTS)
      },
      **{
          'hparams_train_%s' % input_id: ChannelParameter(
              type=standard_artifacts.HyperParameters, optional=True)
          for input_id in range(executor.MAX_INPUTS)
      }
  }
  OUTPUTS = {
      'metamodel':
          ChannelParameter(type=standard_artifacts.Model),
      'output_hyperparameters':
          ChannelParameter(type=artifacts.KCandidateHyperParameters),
  }


class MetaLearner(base_component.BaseComponent):
  """MetaLearner that recommends a tuner config."""

  SPEC_CLASS = MetaLearnerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.MetaLearnerExecutor)

  def __init__(self,
               algorithm: str,
               custom_config: Optional[Dict[str, Any]] = None,
               **meta_train_data: types.Channel):
    """Construct a MetaLearner component.

    Args:
      algorithm: The MetaLearning algorithm to use.
      custom_config: MetaLearning algorithm specific options.
      **meta_train_data: Expected to have the following keys:
       - 'hparams_train_1': Output Channel of Tuner of train dataset 1,
       - 'meta_train_features_1': Output Channel of MetaFeatureGen of train
         dataset 1, . . .
       - 'hparams_train_N': Output Channel of Tuner of train dataset N,
       - 'meta_train_features_N': Output Channel of MetaFeatureGen of train
         dataset N, }

    Raises:
      ValueError: If meta_train_data is not present.
    """

    if not meta_train_data:
      raise ValueError('meta_train_data cannot be empty.')

    model = types.Channel(
        type=standard_artifacts.Model, artifacts=[standard_artifacts.Model()])
    output_hyperparameters = types.Channel(
        type=artifacts.KCandidateHyperParameters,
        artifacts=[artifacts.KCandidateHyperParameters()])
    spec = MetaLearnerSpec(
        algorithm=algorithm,
        metamodel=model,
        output_hyperparameters=output_hyperparameters,
        custom_config=custom_config,
        **meta_train_data)
    super(MetaLearner, self).__init__(spec=spec)
