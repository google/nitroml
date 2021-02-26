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
r"""A model quality benchmark task."""

import abc
from typing import List, Optional, Union

from nitroml.benchmark.result_publisher.component import BenchmarkResultPublisher
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx import components as tfx
from tfx import types
from tfx.dsl.components.base import base_component

from nitroml.protos import problem_statement_pb2 as ps_pb2


class BenchmarkTask(abc.ABC):
  r"""Defines a BenchmarkTask for AutoML."""

  @abc.abstractproperty
  def name(self) -> str:
    """Returns the task's name."""

  @abc.abstractproperty
  def components(self) -> List[base_component.BaseComponent]:
    """Returns TFX components required for the task."""

  @abc.abstractproperty
  def train_and_eval_examples(self) -> types.Channel:
    """Returns train and eval labeled examples."""

  @abc.abstractproperty
  def problem_statement(self) -> ps_pb2.ProblemStatement:
    """Returns the ProblemStatement associated with this BenchmarkTask."""

  @property
  def label_key(self) -> str:
    """Returns the label key from the problem statement."""

    task_type = self.problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return task_type.multi_class_classification.label
    if task_type.HasField('binary_classification'):
      return task_type.binary_classification.label
    if task_type.HasField('one_dimensional_regression'):
      return task_type.one_dimensional_regression.label
    raise ValueError('Invalid task type: {}'.format(task_type))

  @abc.abstractproperty
  def test_examples(self) -> types.Channel:
    """Returns the test labeled examples to use in `make_evaluation`."""

  @abc.abstractproperty
  def test_example_splits(self) -> List[str]:
    """Returns the names of the example splits to use in `make_evaluation`."""

  @property
  def evaluation_metrics(
      self) -> List[Union[tf.keras.metrics.Metric, tfma.metrics.Metric]]:
    """Returns the list of metrics to use in `make_evaluation`."""

    task_type = self.problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return [
          tfma.metrics.ExampleCount(),
          tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
          tf.keras.metrics.SparseCategoricalCrossentropy(name='average_loss')
      ]
    if task_type.HasField('binary_classification'):
      return [
          tfma.metrics.ExampleCount(),
          tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
          tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
      ]
    if task_type.HasField('one_dimensional_regression'):
      return [
          tfma.metrics.ExampleCount(),
          tf.keras.metrics.MeanSquaredError(name='mse'),
      ]
    raise ValueError('Invalid task type: {}'.format(task_type))

  @property
  def prediction_column_key(self) -> str:
    """Returns the column name key for storing predictions in tf.Examples."""

    return 'nitroml_prediction'

  @property
  def ignorable_features(self) -> List[str]:
    """A list of features in the tf.Examples to ignore during training.

    For example, if there is an opaque ID stored as a feature in the tf.Examples
    and used to uniquely identify examples, the feature should be ignored during
    training.
    """

    return []

  def make_evaluation(
      self,
      benchmark_name: str,
      model: Optional[types.Channel] = None,
      predictions: Optional[types.Channel] = None,
      benchmark_run: int = 1,
      runs_per_benchmark: int = 1,
      **kwargs,
  ) -> List[base_component.BaseComponent]:
    """Returns a list of components for evaluating the model.

    Evaluates the model on the 'test' split using the TFX Evaluator, and
    publishes results to MLMD for analysis from the results Colab notebook.

    Args:
      benchmark_name: Unique name of the benchmark, used for publishing
        evalution results.
      model: A `standard_artifacts.Model` Channel, usually produced by a Trainer
        component. Exactly one of `model` and `predictions` must be set.
      predictions: A `standard_artifacts.Examples` Channel, containing
        predictions for `test_examples` in a features column defined by
        `prediction_column_key`. This is generally used for evaluating a model
        that cannot be exported as a `tf.SavedModel`, using its predictions on
        the test set instead. Exactly one of `model` and `predictions` must be
        set.
      benchmark_run: Index of the benchmark run for when runs_per_benchmark > 1.
      runs_per_benchmark: Integer number of runs_per_benchmark. Used for
        computing means and variance for benchmark runs.
      **kwargs: Additional kwargs to pass to `Task#make_evaluation`.

    Raises:
      ValueError: when both `model` and `predictions` are specified.
      ValueError: when neither `model` nor `predictions` are specified.
    """

    del kwargs  # Unused.
    if not model and not predictions:
      raise ValueError(
          'At least one of `model` or `predictions` should be specified')
    if model and predictions:
      raise ValueError(
          'Only one of `model` or `predictions` should be specified')
    metrics_specs = tfma.metrics.specs_from_metrics(
        metrics=self.evaluation_metrics)
    if model:
      evaluator = tfx.Evaluator(
          examples=self.test_examples,
          model=model,
          eval_config=tfma.EvalConfig(
              model_specs=[
                  tfma.ModelSpec(
                      label_key=self.label_key, model_type='tf_generic'),
              ],
              metrics_specs=metrics_specs,
              slicing_specs=[]),
          example_splits=self.test_example_splits,
          instance_name='model')

      publisher = BenchmarkResultPublisher(
          benchmark_name,
          evaluator.outputs.evaluation,
          run=benchmark_run,
          num_runs=runs_per_benchmark,
          instance_name='model')
    else:
      evaluator = tfx.Evaluator(
          examples=predictions,
          eval_config=tfma.EvalConfig(
              model_specs=[
                  tfma.ModelSpec(
                      label_key=self.label_key,
                      prediction_key=self.prediction_column_key),
              ],
              metrics_specs=metrics_specs,
              slicing_specs=[]),
          example_splits=self.test_example_splits,
          instance_name='predictions')

      publisher = BenchmarkResultPublisher(
          benchmark_name,
          evaluator.outputs.evaluation,
          run=benchmark_run,
          num_runs=runs_per_benchmark,
          instance_name='predictions')

    return [evaluator, publisher]
