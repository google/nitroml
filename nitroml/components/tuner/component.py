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
"""The Tuner component for hyperparam tuning, which also outputs trial plot data."""

from typing import Any, Dict, Optional, Text, NamedTuple
from kerastuner.engine import base_tuner

from nitroml.components.tuner import executor
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import TunerSpec
from tfx.components.tuner.component import TunerFnResult
from tfx.utils import json_utils
from tfx.types.artifact import Artifact
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class TunerData(Artifact):
  """NitroML's custom Artifact to store Tuner data for plotting."""

  TYPE_NAME = 'NitroML.TunerData'


class TunerSpec(ComponentSpec):
  """ComponentSpec for custom Tuner Component which saves trial plot data."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, str), optional=True),
      'tuner_fn': ExecutionParameter(type=(str, str), optional=True),
      'train_args': ExecutionParameter(type=trainer_pb2.TrainArgs),
      'eval_args': ExecutionParameter(type=trainer_pb2.EvalArgs),
      'tune_args': ExecutionParameter(type=tuner_pb2.TuneArgs, optional=True),
      'custom_config': ExecutionParameter(type=(str, Any), optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'schema':
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
      'transform_graph':
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
  }
  OUTPUTS = {
      'best_hyperparameters':
          ChannelParameter(type=standard_artifacts.HyperParameters),
      'trial_summary_plot':
          ChannelParameter(type=TunerData),
  }


class Tuner(base_component.BaseComponent):
  """A custom TFX component for model hyperparameter tuning."""

  SPEC_CLASS = TunerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               examples: types.Channel = None,
               schema: Optional[types.Channel] = None,
               transform_graph: Optional[types.Channel] = None,
               module_file: Optional[str] = None,
               tuner_fn: Optional[str] = None,
               train_args: trainer_pb2.TrainArgs = None,
               eval_args: trainer_pb2.EvalArgs = None,
               tune_args: Optional[tuner_pb2.TuneArgs] = None,
               custom_config: Optional[Dict[str, Any]] = None,
               best_hyperparameters: Optional[types.Channel] = None,
               instance_name: Optional[str] = None):
    """Constructs custom Tuner component that stores trial learning curve.

      Adapted from the following code:
      https://github.com/tensorflow/tfx/blob/master/tfx/components/tuner/component.py
    """

    if bool(module_file) == bool(tuner_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'tuner_fn' must be supplied")

    best_hyperparameters = best_hyperparameters or types.Channel(
        type=standard_artifacts.HyperParameters,
        artifacts=[standard_artifacts.HyperParameters()])
    trial_summary_plot = types.Channel(type=TunerData, artifacts=[TunerData()])
    spec = TunerSpec(
        examples=examples,
        schema=schema,
        transform_graph=transform_graph,
        module_file=module_file,
        tuner_fn=tuner_fn,
        train_args=train_args,
        eval_args=eval_args,
        tune_args=tune_args,
        best_hyperparameters=best_hyperparameters,
        trial_summary_plot=trial_summary_plot,
        custom_config=json_utils.dumps(custom_config),
    )
    super(Tuner, self).__init__(spec=spec, instance_name=instance_name)
