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
"""Implements the Ensemble Selection algorithm."""

import collections
from typing import Any, Callable, Dict, List, Optional

from absl import logging
import numpy as np
import tensorflow as tf

from nitroml.protos import problem_statement_pb2 as ps_pb2

LoadedSavedModel = Any
LoadedModelPredictFn = Callable[[LoadedSavedModel, tf.Tensor], tf.Tensor]


def _default_predict_fn(loaded_model, x):
  """Default predict function (callable) for running inference."""
  predictions_dict = loaded_model.signatures['serving_default'](x)
  keys = list(predictions_dict.keys())
  if len(keys) == 1:
    return predictions_dict[keys[0]]
  if 'scores' in predictions_dict:
    return predictions_dict['scores']
  raise ValueError(
      'The predictions dict must be either a single entry or contain a scores key.'
  )


class EnsembleSelection(object):
  """Implements the Ensemble Selection algorithm by Caruana et al."""

  def __init__(self,
               problem_statement: ps_pb2.ProblemStatement,
               saved_model_paths: Dict[str, str],
               ensemble_size: int,
               metric: tf.keras.metrics.Metric,
               predict_fn: Optional[LoadedModelPredictFn] = None,
               goal: Optional[str] = 'maximize'):
    # pyformat: disable
    """Construct an Ensemble Selection object.

    Args:
      problem_statement: ProblemStatement proto identifying the task.
      saved_model_paths: A dict mapping model ids (strings) to SavedModel paths.
      ensemble_size: Maximum number of models (with replacement) to select. This
        is the number of rounds (iterations) for which the ensemble selection
        algorithm will run. The number of models in the final ensemble will be
        at most ensemble_size.
      metric: The TF Keras Metric to optimize for during ensemble selection.
      predict_fn: A function (callable) that maps an input (tf.Tensor) to a
        prediction (tf.Tensor). In addition to the input tf.Tensor, the callable
        takes as argument the (loaded) model that generates the prediction.
        Concretely, predict_fn follows the signature:
        * `loaded_model`: The (loaded) model used for the prediction.
        * `x`: The input (tf.Tensor) to the model.
      goal: String 'maximize' or 'minimize' depending on the goal of the metric.

    Raises:
      ValueError: When goal not 'maximize' or 'minimize'.
    """
    # pyformat: enable
    if goal not in ['maximize', 'minimize']:
      raise ValueError('goal must be either \'maximize\' or \'minimize\'')

    self._problem_statement = problem_statement
    self._saved_model_paths = saved_model_paths
    self._ensemble_size = ensemble_size
    self._metric = metric
    self._predict_fn = predict_fn or _default_predict_fn
    self._goal = goal
    self._weights = {}

  @property
  def weights(self) -> Dict[str, float]:
    return self._weights

  def fit(self, x: np.ndarray, y: np.ndarray):
    """Ensemble selection method by Caruana et al.

    Args:
      x: Numpy array of serialized TF Examples used to make predictions.
      y: Numpy array of labels.
    """
    model_id_to_predictions = self._get_predictions_dict(x)
    ensemble_prediction = np.array(0)
    ensemble_count = collections.defaultdict(int)
    best_ensemble_count = collections.defaultdict(int)
    best_score_overall = -np.inf if self._goal == 'maximize' else np.inf
    for i in range(self._ensemble_size):
      best_score = -np.inf if self._goal == 'maximize' else np.inf
      best_model = None
      for model_id, model_prediction in sorted(model_id_to_predictions.items()):
        updated_ensemble_prediction = self._calculate_ensemble_prediction(
            i, ensemble_prediction, model_prediction)
        self._metric.reset_states()
        score = self._metric(
            y, self._apply_activation(updated_ensemble_prediction))
        if self._score_improved(score, best_score):
          best_score = score
          best_model = model_id
      ensemble_prediction = self._calculate_ensemble_prediction(
          i, ensemble_prediction, model_id_to_predictions[best_model])
      ensemble_count[best_model] += 1
      if self._score_improved(best_score, best_score_overall):
        best_score_overall = best_score
        best_ensemble_count = ensemble_count.copy()
      logging.info('round %d, best score: %s', i, best_score)
    self._calculate_weights(best_ensemble_count)

  def predict(self, x: np.ndarray) -> np.ndarray:
    """Given serialized tf.Examples, compute predictions.

    Args:
      x: Numpy array of serialized TF Examples used to make predictions.

    Returns:
      Numpy array of predictions based on weighted combination of base-model
      predictions determined in `fit`.

    Raises:
      ValueError if called before `fit`.
    """
    if not self.weights:
      raise ValueError(
          'Weights cannot be empty. Must call `fit` before `predict`.')

    return self._get_predictions(x)

  def evaluate(self, x: np.ndarray, y: np.ndarray,
               metrics: List[tf.keras.metrics.Metric]) -> List[float]:
    """Given serialized tf.Examples and labels and metrics, compute evaluations.

    Args:
      x: Numpy 1D array of serialized TF Examples used to make predictions.
      y: Numpy array of labels whose shape either matches or can be broadcast to
        the shape of the ensemble predictions.
      metrics: List of TF Keras Metrics

    Returns:
      A list containing each computed metric value as a float  in the same order
      as they appear in the input `metrics` list.

    Raises:
      ValueError if called before `fit`.
    """
    if not self.weights:
      raise ValueError(
          'Weights cannot be empty. Must call `fit` before `evaluate`.')

    ensemble_prediction = self.predict(x)
    return [metric(y, ensemble_prediction) for metric in metrics]

  def save(self, export_path: str) -> str:
    """Saves ensemble to disk as a SavedModel to be used for serving.

    Args:
      export_path (str): The directory where the combined SavedModel will be
        exported.

    Returns:
      A string containing the path to the saved model.
    """
    return self._save_combined_model(export_path)

  def _get_predictions_dict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
    """Returns predictions for every model loaded from saved_model_paths.

    Args:
      x: Numpy array of serialized TF Examples used to make predictions.
    """
    predictions = {}
    for model_id, path in self._saved_model_paths.items():
      reloaded_model = tf.saved_model.load(path)
      predictions[model_id] = self._predict_fn(reloaded_model,
                                               tf.constant(x)).numpy()
    return predictions

  def _score_improved(self, score: float, best_score: float) -> bool:
    """Returns if the score improved from the previous best score.

    Args:
      score: The new score as a float.
      best_score: Previous best score to compare against.
    """
    if self._goal == 'maximize':
      return score > best_score
    return score < best_score

  def _calculate_ensemble_prediction(
      self, round_num: int, prev_prediction: np.ndarray,
      model_prediction: np.ndarray) -> np.ndarray:
    """Returns the new ensemble prediction by computing the streaming mean.

    Args:
      round_num: The round number, or iteration number, of the ensemble.
      prev_prediction: The ensemble prediction prior to adding the new model.
      model_prediction: Prediction values of the added model.
    """
    return (round_num * prev_prediction + model_prediction) / (round_num + 1)

  def _calculate_weights(self, ensemble_count: Dict[str, int]):
    """Updates weights from the results of fit.

    Args:
      ensemble_count: A dict mapping model id strings to its count in the
        ensemble.
    """
    total_count = sum(ensemble_count.values())
    for model_id, count in ensemble_count.items():
      self._weights[model_id] = count / total_count

  def _apply_activation(self, predictions):
    """Returns the Estimator Head for this task."""

    task_type = self._problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return tf.nn.softmax(predictions)
    if task_type.HasField('binary_classification'):
      return tf.nn.softmax(predictions)[:, 1]
    return predictions

  def _get_predictions(self, x: np.ndarray) -> np.ndarray:
    predictions = np.zeros(shape=(len(x), 1))
    for model_id, weight in self.weights.items():
      reloaded_model = tf.saved_model.load(self._saved_model_paths[model_id])
      model_prediction = self._predict_fn(reloaded_model,
                                          tf.convert_to_tensor(x)).numpy()
      predictions = predictions + model_prediction * weight
    return predictions

  def _get_weighted_prediction(self, loaded_models: Dict[str, LoadedSavedModel],
                               examples: tf.Tensor) -> tf.Tensor:
    """Returns the weighted prediction for the provided example.

    Args:
      loaded_models: A dict of model_id to loaded saved models.
      examples: A tensor of Examples used to make predictions.

    Returns:
      The weighted predictions (output) for the provided examples
    """
    weighted_prediction = tf.constant(0.0)
    for model_id, weight in self.weights.items():
      loaded_model = loaded_models[model_id]
      weighted_prediction = weighted_prediction + self._predict_fn(
          loaded_model, examples) * weight
    return self._apply_activation(weighted_prediction)

  def _save_combined_model(self, export_path: str) -> str:
    """Saves a weighted combination of saved models as a standalone SavedModel.

    The weighted combination refers to the final output (prediction) of the
    combined model. For every input, the output of the combined model is the
    weighted (as specified by weights) sum of the outputs of the input saved
    models.

    Args:
      export_path (str): The directory where the combined SavedModel will be
        exported.

    Returns:
      A string containing the path to the combined saved model.
    """

    def _model_fn(features: Dict[str, tf.Tensor], labels: tf.Tensor,
                  mode: tf.estimator.ModeKeys) -> tf.estimator.EstimatorSpec:
      """Model function passed to tf.estimator.Estimator.

      Args:
        features: Examples to be used to make predictions.
        labels: The labels for the given Examples.
        mode: Specifies if this is training, evaluation or prediction.

      Returns:
        A tf.estimator.EstimatorSpec defining the model to be run by an
        Estimator.
      """
      del labels

      loaded_models = {
          model_id: tf.saved_model.load(self._saved_model_paths[model_id])
          for model_id in self.weights.keys()
      }

      # Train with no-op so that the Estimator doesn't throw an exception.
      if mode == tf.estimator.ModeKeys.TRAIN:
        step = tf.compat.v1.train.get_global_step()
        train_op = step.assign_add(1)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=tf.constant([0]), train_op=train_op)

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = self._get_weighted_prediction(loaded_models,
                                                    features['example'])
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, train_op=None)

    def _serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.compat.v1.placeholder(
          dtype=tf.string, shape=(None), name='serialized_example')
      return tf.estimator.export.ServingInputReceiver(
          features={'example': serialized_example},
          receiver_tensors=serialized_example)

    # Train for 1 step so that the Estimator doesn't throw an exception.
    estimator = tf.estimator.Estimator(_model_fn)
    estimator.train(
        input_fn=lambda: (tf.constant('empty'), tf.no_op()), max_steps=1)
    saved_path = estimator.export_saved_model(export_path, _serving_input_fn)
    return saved_path
