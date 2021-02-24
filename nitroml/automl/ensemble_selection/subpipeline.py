# Copyright 2021 Google LLC
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
"""An Ensemble Selection subpipeline for tabular datasets."""

import json
import os
from typing import List, Optional, Tuple

from absl import logging
from nitroml import subpipeline
from nitroml.automl.ensemble_selection.lib import ensemble_selection as es_lib
import tensorflow as tf
from tfx import types
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components.base import base_component
from tfx.types import standard_artifacts
from tfx.utils import path_utils

from google.protobuf import text_format
from nitroml.protos import problem_statement_pb2 as ps_pb2


class EnsembleSelection(subpipeline.Subpipeline):
  """An Ensemble Selection subpipeline for tabular datasets."""

  def __init__(self,
               problem_statement: ps_pb2.ProblemStatement,
               examples: types.Channel,
               models: List[types.Channel],
               evaluation_split_name: str,
               ensemble_size: int,
               metric: Optional[tf.keras.metrics.Metric] = None,
               goal: Optional[str] = None,
               instance_name: Optional[str] = None):
    """Constructs an AutoTrainer subpipeline.

    Args:
      problem_statement: ProblemStatement proto identifying the task.
      examples: A Channel of 'Example' artifact type produced from an upstream
        The source of examples that are used in evaluation (required).
      models: A List of Channels of 'standard_artifact.Model' type to use as
        the library of base models in the ensemble selection algorithm.
      evaluation_split_name: String name of the evaluation split in the
        `examples` artifact to use for evaluation. For examples, 'eval'.
      ensemble_size: Maximum number of models (with replacement) to select. This
        is the number of rounds (iterations) for which the ensemble selection
        algorithm will run. The number of models in the final ensemble will be
        at most ensemble_size.
      metric: Optional TF Keras Metric to optimize for during ensemble
        selection. When `None`, the `problem_statement` is used to determine
        the metric and goal.
      goal: Optional string 'maximize' or 'minimize' depending on the goal of
        the metric. When `None`, the `problem_statement` is used to determine
        the metric and goal.
      instance_name: Optional unique instance name. Necessary iff multiple
        EnsembleSelection subpipelines are declared in the same pipeline.

    Raises:
      ValueError: When a required param is not supplied.
    """

    if not metric and not goal:
      metric, goal = self._determine_metric_and_goal(problem_statement)

    input_models = {f'input_model{i}': model for i, model in enumerate(models)}

    self._instance_name = instance_name
    self._ensemble_selection = ensemble_selection(
        problem_statement=text_format.MessageToString(
            message=problem_statement, as_utf8=True),
        examples=examples,
        evaluation_split_name=evaluation_split_name,
        ensemble_size=ensemble_size,
        metric=json.dumps(tf.keras.metrics.serialize(metric)),
        goal=goal,
        instance_name=instance_name,
        **input_models,
    )

  @property
  def id(self) -> str:
    """Returns the AutoTrainer sub-pipeline's unique ID."""

    autotrainer_instance_name = 'EnsembleSelection'
    if self._instance_name:
      autotrainer_instance_name = f'{autotrainer_instance_name}.{self._instance_name}'
    return autotrainer_instance_name

  @property
  def components(self) -> List[base_component.BaseComponent]:
    """Returns the AutoTrainer sub-pipeline's constituent components."""

    return [self._ensemble_selection]

  @property
  def outputs(self) -> subpipeline.SubpipelineOutputs:
    """Return the AutoTrainer sub-pipeline's outputs."""

    return subpipeline.SubpipelineOutputs(
        {'model': self._ensemble_selection.outputs.model})

  def _determine_metric_and_goal(
      self, problem_statement: ps_pb2.ProblemStatement
  ) -> Tuple[tf.keras.metrics.Metric, str]:
    task_type = problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return tf.keras.metrics.SparseCategoricalAccuracy(
          name='accuracy'), 'maximize'
    if task_type.HasField('binary_classification'):
      return tf.keras.metrics.AUC(name='auc_roc', curve='ROC'), 'maximize'
    if task_type.HasField('one_dimensional_regression'):
      return tf.keras.metrics.MeanSquaredError(name='mse'), 'minimize'
    raise ValueError('Invalid task type: {}'.format(task_type))


# pytype: disable=wrong-arg-types
@component
def ensemble_selection(
    problem_statement: Parameter[str],
    examples: InputArtifact[standard_artifacts.Examples],
    evaluation_split_name: Parameter[str],
    ensemble_size: Parameter[int],
    metric: Parameter[str],
    goal: Parameter[str],
    model: OutputArtifact[standard_artifacts.Model],
    input_model0: InputArtifact[standard_artifacts.Model] = None,
    input_model1: InputArtifact[standard_artifacts.Model] = None,
    input_model2: InputArtifact[standard_artifacts.Model] = None,
    input_model3: InputArtifact[standard_artifacts.Model] = None,
    input_model4: InputArtifact[standard_artifacts.Model] = None,
    input_model5: InputArtifact[standard_artifacts.Model] = None,
    input_model6: InputArtifact[standard_artifacts.Model] = None,
    input_model7: InputArtifact[standard_artifacts.Model] = None,
    input_model8: InputArtifact[standard_artifacts.Model] = None,
    input_model9: InputArtifact[standard_artifacts.Model] = None,
) -> None:  # pytype: disable=invalid-annotation,wrong-arg-types
  """Runs the SimpleML trainer as a separate component."""

  problem_statement = text_format.Parse(problem_statement,
                                        ps_pb2.ProblemStatement())
  input_models = [
      input_model0, input_model1, input_model2, input_model3, input_model4,
      input_model5, input_model6, input_model7, input_model8, input_model9
  ]
  saved_model_paths = {
      str(i): path_utils.serving_model_path(model.uri)
      for i, model in enumerate(input_models)
      if model
  }
  logging.info('Saved model paths: %s', saved_model_paths)

  label_key = _label_key(problem_statement)

  es = es_lib.EnsembleSelection(
      problem_statement=problem_statement,
      saved_model_paths=saved_model_paths,
      ensemble_size=ensemble_size,
      metric=tf.keras.metrics.deserialize(json.loads(metric)),
      goal=goal)

  es.fit(*_data_from_examples(
      examples_path=os.path.join(examples.uri, evaluation_split_name),
      label_key=label_key))
  logging.info('Selected ensemble weights: %s', es.weights)
  es.save(
      export_path=os.path.join(
          path_utils.serving_model_dir(model.uri), 'export', 'serving'))


# pytype: enable=wrong-arg-types


def _data_from_examples(examples_path: str, label_key: str):
  """Returns a tuple of ndarrays of examples and label values."""

  # Load all the examples.
  filenames = tf.io.gfile.listdir(examples_path)
  files = [
      os.path.join(examples_path, filename) for filename in sorted(filenames)
  ]
  dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
  x, y = [], []
  for serialized_example in dataset.take(10000).as_numpy_iterator():
    x.append(serialized_example)
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    y.append(_label_value(example, label_key))
  return x, y


def _label_key(problem_statement: ps_pb2.ProblemStatement) -> str:
  """Returns the label key from the problem statement."""

  task_type = problem_statement.tasks[0].type
  if task_type.HasField('multi_class_classification'):
    return task_type.multi_class_classification.label
  if task_type.HasField('binary_classification'):
    return task_type.binary_classification.label
  if task_type.HasField('one_dimensional_regression'):
    return task_type.one_dimensional_regression.label
  raise ValueError('Invalid task type: {}'.format(task_type))


def _label_value(example: tf.train.Example, label_key: str):
  feature = example.features.feature[label_key]
  if feature.HasField('int64_list'):
    return feature.int64_list.value
  if feature.HasField('float_list'):
    return feature.float_list.value
  return feature.bytes_list.value
