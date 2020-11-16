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

from typing import Any, Dict, Optional

from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.tuner import executor
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.utils import json_utils


class TunerData(Artifact):
  """NitroML's custom Artifact to store Tuner data for plotting."""

  TYPE_NAME = 'NitroML.TunerData'


class AugmentedTunerSpec(ComponentSpec):
  """Component spec for AugmentedTuner which also saves the trial plot data."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, str), optional=True),
      'tuner_fn': ExecutionParameter(type=(str, str), optional=True),
      'train_args': ExecutionParameter(type=trainer_pb2.TrainArgs),
      'eval_args': ExecutionParameter(type=trainer_pb2.EvalArgs),
      'tune_args': ExecutionParameter(type=tuner_pb2.TuneArgs, optional=True),
      'custom_config': ExecutionParameter(type=(str, Any), optional=True),
      'metalearning_algorithm': ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'schema':
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
      'transform_graph':
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
      'warmup_hyperparameters':
          ChannelParameter(
              type=artifacts.KCandidateHyperParameters, optional=True),
      'metamodel':
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      'metafeature':
          ChannelParameter(type=artifacts.MetaFeatures, optional=True),
  }
  OUTPUTS = {
      'best_hyperparameters':
          ChannelParameter(type=standard_artifacts.HyperParameters),
      'trial_summary_plot':
          ChannelParameter(type=TunerData),
  }


# TODO(nikhilmehta): Find a better way to use existing tfx.Tuner implementation.
# Currently, inheritance isn't viable since tfx.Tuner doesn't accept custom
# spec.
class AugmentedTuner(base_component.BaseComponent):
  """A custom TFX component for model hyperparameter tuning."""

  SPEC_CLASS = AugmentedTunerSpec
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
               metalearning_algorithm: Optional[str] = None,
               warmup_hyperparameters: Optional[types.Channel] = None,
               metamodel: Optional[types.Channel] = None,
               metafeature: Optional[types.Channel] = None,
               best_hyperparameters: Optional[types.Channel] = None,
               instance_name: Optional[str] = None):
    """Constructs custom Tuner component that stores trial learning curve.

    Adapted from the following code:
    https://github.com/tensorflow/tfx/blob/master/tfx/components/tuner/component.py

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, serving as the
        source of examples that are used in tuning (required).
      schema:  An optional Channel of type `standard_artifacts.Schema`, serving
        as the schema of training and eval data. This is used when raw examples
        are provided.
      transform_graph: An optional Channel of type
        `standard_artifacts.TransformGraph`, serving as the input transform
        graph if present. This is used when transformed examples are provided.
      module_file: A path to python module file containing UDF tuner definition.
        The module_file must implement a function named `tuner_fn` at its top
        level. The function must have the following signature.
            def tuner_fn(fn_args: FnArgs) -> TunerFnResult: Exactly one of
              'module_file' or 'tuner_fn' must be supplied.
      tuner_fn:  A python path to UDF model definition function. See
        'module_file' for the required signature of the UDF. Exactly one of
        'module_file' or 'tuner_fn' must be supplied.
      train_args: A trainer_pb2.TrainArgs instance, containing args used for
        training. Currently only splits and num_steps are available. Default
        behavior (when splits is empty) is train on `train` split.
      eval_args: A trainer_pb2.EvalArgs instance, containing args used for eval.
        Currently only splits and num_steps are available. Default behavior
        (when splits is empty) is evaluate on `eval` split.
      tune_args: A tuner_pb2.TuneArgs instance, containing args used for tuning.
        Currently only num_parallel_trials is available.
      custom_config: A dict which contains addtional training job parameters
        that will be passed into user module.
      metalearning_algorithm: Optional str for the type of
        metalearning_algorithm.
      warmup_hyperparameters: Optional Channel of type
        `artifacts.KCandidateHyperParameters` for a list of recommended search
        space for warm-starting the tuner (generally the output of a
        metalearning component or subpipeline).
      metamodel: Optional Channel of type `standard_artifacts.Model` for trained
        meta model
      metafeature: Optional Channel of `artifacts.MetaFeatures` of the dataset
        to be tuned. This is used as an input to the `meta_model` to predict
        search space.
      best_hyperparameters: Optional Channel of type
        `standard_artifacts.HyperParameters` for result of the best hparams.
      instance_name: Optional unique instance name. Necessary if multiple Tuner
        components are declared in the same pipeline.
    """

    if bool(module_file) == bool(tuner_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'tuner_fn' must be supplied")

    best_hyperparameters = best_hyperparameters or types.Channel(
        type=standard_artifacts.HyperParameters,
        artifacts=[standard_artifacts.HyperParameters()])
    trial_summary_plot = types.Channel(type=TunerData, artifacts=[TunerData()])
    spec = AugmentedTunerSpec(
        examples=examples,
        schema=schema,
        transform_graph=transform_graph,
        module_file=module_file,
        tuner_fn=tuner_fn,
        train_args=train_args,
        eval_args=eval_args,
        tune_args=tune_args,
        metalearning_algorithm=metalearning_algorithm,
        warmup_hyperparameters=warmup_hyperparameters,
        metamodel=metamodel,
        metafeature=metafeature,
        best_hyperparameters=best_hyperparameters,
        trial_summary_plot=trial_summary_plot,
        custom_config=json_utils.dumps(custom_config),
    )
    super(AugmentedTuner, self).__init__(spec=spec, instance_name=instance_name)
