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
from typing import Any, Dict, List

from absl import logging
import numpy as np
import tensorflow as tf

LoadedSavedModel = Any


class EnsembleSelection(object):
  """Implements the Ensemble Selection algorithm by Caruana et al."""

  def __init__(self,
               saved_model_paths: Dict[str, str],
               ensemble_size: int,
               metric: tf.keras.metrics.Metric,
               goal: str = 'maximize'):
    """Construct an Ensemble Selection object.

    Args:
      saved_model_paths: A dict mapping model id strings to its SavedModel path.
      ensemble_size: Maximum number of ensembles to select.
      metric: The TF Keras Metric to optimize for during ensemble selection.
      goal: String 'maximize' or 'minimize' depending on the goal of the metric.

    Raises:
      ValueError: When goal not 'maximize' or 'minimize'.
    """

    if goal not in ['maximize', 'minimize']:
      raise ValueError('goal must be either \'maximize\' or \'minimize\'')

    self._saved_model_paths = saved_model_paths
    self._ensemble_size = ensemble_size
    self._metric = metric
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
    predictions = self._get_predictions_dict(x)
    ensemble_prediction = np.zeros(shape=len(self._saved_model_paths))
    ensemble_count = collections.defaultdict(int)
    best_ensemble_count = collections.defaultdict(int)
    best_score_overall = float('-inf') if self._goal == 'maximum' else float(
        'inf')
    for i in range(self._ensemble_size):
      best_score = float('-inf') if self._goal == 'maximum' else float('inf')
      best_id = None
      for model_id, model_prediction in predictions.items():
        new_prediction = self._calculate_ensemble_prediction(
            i, ensemble_prediction, model_prediction)
        self._metric.reset_states()
        score = self._metric(y, new_prediction)
        if self._score_improved(score, best_score):
          best_score = score
          best_id = model_id
      ensemble_prediction = self._calculate_ensemble_prediction(
          i, ensemble_prediction, predictions[best_id])
      ensemble_count[best_id] += 1
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

    predictions = np.zeros(shape=(len(x), 1))
    for model_id, weight in self.weights.items():
      reloaded_model = tf.saved_model.load(self._saved_model_paths[model_id])
      model_prediction = reloaded_model.signatures['serving_default'](
          tf.convert_to_tensor(x))['output_0'].numpy()
      predictions = predictions + model_prediction * weight
    return predictions

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
      export_path: A string directory used to export SavedModel.

    Returns:
      A string containing the path to the saved model.
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

      nonzero_models = {
          model_id: tf.saved_model.load(self._saved_model_paths[model_id])
          for model_id in self.weights.keys()
      }

      # train with no-op so that Estimator doesn't throw excetion
      if mode == tf.estimator.ModeKeys.TRAIN:
        step = tf.compat.v1.train.get_global_step()
        train_op = step.assign_add(1)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=tf.constant([0]), train_op=train_op)

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = self._get_weighted_prediction(nonzero_models,
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

    # train for 1 step so that Estimator doesn't throw exception
    estimator = tf.estimator.Estimator(_model_fn)
    estimator.train(
        input_fn=lambda: (tf.constant('empty'), tf.no_op()), max_steps=1)
    saved_path = estimator.export_saved_model(export_path, _serving_input_fn)
    return saved_path

  def _get_weighted_prediction(self, models: Dict[str, LoadedSavedModel],
                               example: tf.Tensor) -> tf.Tensor:
    """Returns the weighted predictions for example as a tensor.

    Args:
      models: A dict of model_id to loaded saved models of non-zero weight.
      example: A tensor of Examples used to make predictions.
    """
    results = tf.constant(0.0)
    for model_id, weight in self.weights.items():
      results = results + models[model_id].signatures['serving_default'](
          examples=example)['output_0'] * weight
    return results

  def _get_predictions_dict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
    """Returns predictions for every model loaded from saved_model_paths.

    Args:
      x: Numpy array of serialized TF Examples used to make predictions.
    """
    predictions = {}
    for model_id, path in self._saved_model_paths.items():
      reloaded_model = tf.saved_model.load(path)
      # TODO(liumich): make configurable depending on model type / serving_fn
      predictions[model_id] = reloaded_model.signatures['serving_default'](
          tf.convert_to_tensor(x))['output_0'].numpy()
    return predictions

  def _score_improved(self, score: float, best_score: float) -> bool:
    """Returns if the score improved from the previous best score.

    Args:
      score: The new score as a float.
      best_score: Previous best score to compare against.
    """
    if self._goal == 'maximize':
      return score >= best_score
    return score <= best_score

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
      weight = count / total_count
      self._weights[model_id] = weight
