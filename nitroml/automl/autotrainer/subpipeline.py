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
"""A self-tuning trainer that produces a model for tabular datasets."""

from typing import List, Optional

from nitroml import subpipeline
from tfx import components as tfx
from tfx import types
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.proto import trainer_pb2

from google.protobuf import text_format
from nitroml.protos import problem_statement_pb2 as ps_pb2


class AutoTrainer(subpipeline.Subpipeline):
  """A self-tuning trainer that produces a model for tabular datasets.

  It is designed to be used in conjunction with NitroML's `AutoData` subpipeline
  using the `BasicPreprocessor`.

  ## Example:
  ```python
  task = MyBenchmarkTask(...)
  autodata = AutoData(
        task.problem_statement,
        examples=task.train_and_eval_examples,
        preprocessor=BasicPreprocessor())
  autotrainer= AutoTrainer(
      problem_statement=task.problem_statement,
      transformed_examples=autodata.transformed_examples,
      transform_graph=autodata.transform_graph,
      schema=autodata.schema,
      train_steps=1000,
      eval_steps=1000)

  pipeline = task.components + autodata.components + autotrainer.components
  ```
  """

  def __init__(self,
               problem_statement: ps_pb2.ProblemStatement,
               transformed_examples: types.Channel,
               transform_graph: types.Channel,
               schema: types.Channel,
               train_steps: int,
               eval_steps: int,
               use_keras: bool = True,
               enable_tuning: bool = False,
               max_sequence_length: Optional[int] = None,
               instance_name: Optional[str] = None):
    """Constructs an AutoTrainer subpipeline.

    Args:
      problem_statement: ProblemStatement proto identifying the task.
      transformed_examples: A Channel of 'ExamplesPath' type produced from an
        upstream Transform component. The source of examples that are used in
        training and evaluation (required).
      transform_graph: An optional Channel of 'TransformPath' type, serving as
        the input transform graph if present.
      schema:  An optional Channel of 'SchemaPath' type, serving as the schema
        of training and eval data.
      train_steps: Number of steps (batches) to train for.
      eval_steps: Number of steps (batches) to evaluate.
      use_keras: When `True`, uses Keras Models, otherwise uses Estimators.
      enable_tuning: When `True`, performs hyperparameter tuning using the
        built-in `tfx.Tuner` using a tuned search-space.
      max_sequence_length: For seqential prediction tasks. When > 0, the
        trainer will produce a model that will produce sequential prediction of
        this desired length.
      instance_name: Optional unique instance name. Necessary iff multiple Tuner
        components are declared in the same pipeline.

    Raises:
      ValueError: When a required param is not supplied.
    """

    self._instance_name = instance_name
    self._tuner = None
    if enable_tuning:
      # Search over search space of model hyperparameters.
      self._tuner = tfx.Tuner(
          tuner_fn='nitroml.automl.autotrainer.lib.auto_trainer.tuner_fn',
          examples=transformed_examples,
          transform_graph=transform_graph,
          train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
          eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
          custom_config={
              # Pass the problem statement proto as a text proto. Required
              # since custom_config must be JSON-serializable.
              'problem_statement':
                  text_format.MessageToString(
                      message=problem_statement, as_utf8=True),
          },
          instance_name=self.id)

    self._trainer = tfx.Trainer(
        run_fn='nitroml.automl.autotrainer.lib.auto_trainer.run_fn' if use_keras
        else 'nitroml.automl.autotrainer.lib.auto_estimator_trainer.run_fn',
        custom_executor_spec=(executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor)),
        transformed_examples=transformed_examples,
        transform_graph=transform_graph,
        schema=schema,
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
        hyperparameters=self._tuner.outputs.best_hyperparameters
        if self._tuner else None,
        custom_config={
            # Pass the problem statement proto as a text proto. Required
            # since custom_config must be JSON-serializable.
            'problem_statement':
                text_format.MessageToString(
                    message=problem_statement, as_utf8=True),
            'sequence_length':
                max_sequence_length,
        },
        instance_name=self.id)

  @property
  def id(self) -> str:
    """Returns the AutoTrainer sub-pipeline's unique ID."""

    autotrainer_instance_name = 'AutoTrainer'
    if self._instance_name:
      autotrainer_instance_name = f'{autotrainer_instance_name}.{self._instance_name}'
    return autotrainer_instance_name

  @property
  def components(self) -> List[base_component.BaseComponent]:
    """Returns the AutoTrainer sub-pipeline's constituent components."""

    return ([self._tuner] if self._tuner else []) + [self._trainer]

  @property
  def outputs(self) -> subpipeline.SubpipelineOutputs:
    """Return the AutoTrainer sub-pipeline's outputs."""

    return subpipeline.SubpipelineOutputs({
        'model':
            self._trainer.outputs.model,
        'best_hyperparameters':
            self._tuner.outputs.best_hyperparameters if self._tuner else None,
    })
