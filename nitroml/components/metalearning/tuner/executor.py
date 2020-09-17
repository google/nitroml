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
"""The AugmentedTuner executor with trial callbacks for tracking trial data."""

import json
import os
import sys
from typing import Any, Dict, List, Tuple, Type, cast

from absl import logging
import kerastuner
from kerastuner.engine import base_tuner
import numpy as np
import tensorflow as tf
from tfx import types
from tfx.components.trainer import fn_args_utils
from tfx.components.tuner import executor as tfx_tuner
from tfx.components.tuner.component import TunerFnResult
from tfx.components.util import udf_utils
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils

DEFAULT_WARMUP_TRIALS = 6
DEFAULT_K = 3
WARMUP_HYPERPARAMETERS = 'warmup_hyperparameters'
CUSTOM_TUNER_NAME = 'get_tuner_cls_with_callbacks.<locals>.TrialTrackingTuner'
BEST_CUMULATIVE_SCORE = 'best_cumulative_score'
OBJECTIVE_DIRECTION = 'objective_direction'


def get_tuner_cls_with_callbacks(tuner_class: Type[base_tuner.BaseTuner]):
  """Returns that custom TrialTrackingTuner class which overrides the trial callbacks.

  Args:
    tuner_class: An existing tuner class that extends the base_tuner.BaseTuner.
  """

  class TrialTrackingTuner(tuner_class):  # pylint: disable=E0239,use-symbolic-message-instead
    """A Tuner which dynamically inherits tuner_class and implements trial callbacks."""

    def on_search_begin(self):
      super(TrialTrackingTuner, self).on_search_begin()
      self._trial_plot_data = {}
      self._trial_plot_data[
          OBJECTIVE_DIRECTION] = self.oracle.objective.direction
      self._trial_plot_data[BEST_CUMULATIVE_SCORE] = []

    def on_trial_end(self, trial):
      super(TrialTrackingTuner, self).on_trial_end(trial)
      best_trial = self.oracle.get_best_trials()[0]
      self._trial_plot_data[BEST_CUMULATIVE_SCORE].append(best_trial.score)

    def get_tuner_plot_data(self) -> Dict[str, Any]:
      return self._trial_plot_data

  return TrialTrackingTuner


def extract_tuner_trial_progress(tuner: base_tuner.BaseTuner) -> Dict[str, Any]:
  """Extract trial progress from the `TrialTrackingTuner` kerastuner.

  Args:
    tuner: The kerastuner of type TrialTrackingTuner.

  Returns:
    A dict of tuner plot data.

  Raises:
    TypeError: When the tuner is not of type TrialTrackingTuner.
  """

  classname = tuner.__class__.__qualname__
  if classname == CUSTOM_TUNER_NAME:
    # Need to cast to Any because get_tuner_plot_data is not part of BaseTuner.
    return cast(Any, tuner).get_tuner_plot_data()
  else:
    raise TypeError(
        'Tuner is expected to have the class '
        f'{CUSTOM_TUNER_NAME}, but got {classname}. '
        'Use `get_tuner_cls_with_callbacks()` to define the kerastuner.')


def merge_trial_data(*all_tuner_data) -> Tuple[Dict[str, Any], int]:
  """Merges sorted trial progress based on objective_score of TrialTrackingTuner tuners.

  The objective_score is based on kerastuner.oracle.Objective.

  Args:
    *all_tuner_data: List of tuner data stored as dictionary.

  Returns:
    The a two-tuple of the cumulative trial data and best performing tuner.

  Raises:
    ValueError: If the list `all_tuner_data` is None or empty.
  """

  if not all_tuner_data:
    raise ValueError('The arg `all_tuner_data` cannot be empty or None.')

  obj_direction = all_tuner_data[0][OBJECTIVE_DIRECTION]
  cmp_fn = max if obj_direction == 'max' else min
  best_score_so_far = sys.float_info.min if obj_direction == 'max' else sys.float_info.max
  best_performing_tuner = 0
  best_tuner_score = []
  cumulative_best_score = []

  for tuner_data in all_tuner_data:
    # sorted score list from the tuner.
    trial_data = tuner_data[BEST_CUMULATIVE_SCORE]
    best_tuner_score.append(trial_data[-1])
    trial_data = [cmp_fn(score, best_score_so_far) for score in trial_data]
    cumulative_best_score.extend(trial_data)
    best_score_so_far = cumulative_best_score[-1]

  cumulative_trial_data = {}
  cumulative_trial_data[BEST_CUMULATIVE_SCORE] = cumulative_best_score
  cumulative_trial_data[OBJECTIVE_DIRECTION] = obj_direction

  # Find the best tuner.
  best_performing_tuner = cmp_fn(
      enumerate(best_tuner_score), key=lambda x: x[1])

  return (cumulative_trial_data, best_performing_tuner[0])


def _load_keras_model(model_path: str):
  """Loads the keras model."""

  model = tf.keras.models.load_model(model_path)
  model.summary()
  return model


def _merge_hparam_configs(configs):
  """Merges Keras HParam configs."""

  configs = [config['values'] for config in configs]
  hparams = configs[0].keys()

  search_space = {}
  for key in hparams:
    search_space[key] = [config[key] for config in configs]

  discrete_search_space = kerastuner.HyperParameters()
  for key, value_list in search_space.items():
    candidate_list = list(set(value_list))
    discrete_search_space.Choice(key, candidate_list)

  return discrete_search_space.get_config()


class Executor(base_executor.BaseExecutor):
  """The executor for nitroml.components.tuner.components.Tuner."""

  def search(self, tuner_fn_result: TunerFnResult) -> base_tuner.BaseTuner:
    """Conduct a single hyperparameter search loop, and return the Tuner."""

    tuner = tuner_fn_result.tuner
    tuner.search_space_summary()
    logging.info('Start tuning... Tuner ID: %s, Max Trials: %d', tuner.tuner_id,
                 tuner.oracle.max_trials)
    tuner.search(**tuner_fn_result.fit_kwargs)
    logging.info('Finished tuning... Tuner ID: %s', tuner.tuner_id)
    tuner.results_summary()

    return tuner

  def warmup(self, input_dict: Dict[str, List[types.Artifact]],
             exec_properties: Dict[str, List[types.Artifact]], algorithm: str):

    # Perform warmup tuning if WARMUP_HYPERPARAMETERS given.
    hparams_warmup_config_list = None
    if input_dict.get(WARMUP_HYPERPARAMETERS):
      hyperparameters_file = io_utils.get_only_uri_in_dir(
          artifact_utils.get_single_uri(input_dict[WARMUP_HYPERPARAMETERS]))
      hparams_warmup_config_list = json.loads(
          io_utils.read_string_file(hyperparameters_file))

    fn_args = fn_args_utils.get_common_fn_args(
        input_dict, exec_properties, working_dir=self._get_tmp_dir() + 'warmup')

    # TODO(nikhilmehta): Currently all algorithms need warmup_hyperparameters.
    # This may not be needed for other algorithms that can predict hyperparams.
    if not hparams_warmup_config_list:
      raise ValueError('Expected warmup_hyperparameters')

    logging.info('Algorithm: %s', algorithm)
    warmup_trials = 0
    if algorithm == 'majority_voting':
      warmup_trials = DEFAULT_WARMUP_TRIALS
      fn_args.custom_config[
          WARMUP_HYPERPARAMETERS] = hparams_warmup_config_list[0]
    elif algorithm == 'nearest_neighbor':
      warmup_trials = DEFAULT_WARMUP_TRIALS

      if input_dict.get('metamodel'):
        metamodel_path = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(input_dict['metamodel']))
        logging.info('Meta model path: %s', metamodel_path)
        metamodel = _load_keras_model(metamodel_path)
      else:
        raise ValueError(
            f'Tuner for metalearning_algorithm={algorithm} expects metamodel.')

      if input_dict.get('metafeature'):
        metafeature_path = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(input_dict['metafeature']))
        logging.info('Metafeature: %s', metafeature_path)
        metafeature = json.loads(io_utils.read_string_file(metafeature_path))
        metafeature = metafeature['metafeature']
      else:
        raise ValueError(
            f'Tuner for metalearning_algorithm={algorithm} expects metafeature.'
        )

      metafeature = np.array(metafeature, dtype=np.float32)
      metafeature = np.expand_dims(metafeature, axis=0)
      logits = metamodel(metafeature).numpy()[0]
      nearest_configs = [
          hparams_warmup_config_list[ix]
          for ix in np.argsort(logits)[-DEFAULT_K:]
      ]
      nearest_hparam_config = _merge_hparam_configs(nearest_configs)
      fn_args.custom_config[WARMUP_HYPERPARAMETERS] = nearest_hparam_config
    else:
      raise NotImplementedError(
          f'Tuning for metalearning_algorithm={algorithm} is not implemented.')

    # kerastuner doesn't support grid search, setting max_trials large enough.
    # Track issue: https://github.com/keras-team/keras-tuner/issues/340
    fn_args.custom_config['max_trials'] = warmup_trials
    tuner_fn = udf_utils.get_fn(exec_properties, 'tuner_fn')
    warmtuner_fn_result = tuner_fn(fn_args)
    warmup_tuner = self.search(warmtuner_fn_result)

    return warmup_tuner, warmup_trials

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:

    if tfx_tuner.get_tune_args(exec_properties):
      raise ValueError("TuneArgs is not supported by this Tuner's Executor.")

    metalearning_algorithm = None
    if 'metalearning_algorithm' in exec_properties:
      metalearning_algorithm = exec_properties.get('metalearning_algorithm')

    warmup_trials = 0
    warmup_trial_data = None
    if metalearning_algorithm:
      warmup_tuner, warmup_trials = self.warmup(input_dict, exec_properties,
                                                metalearning_algorithm)
      warmup_trial_data = extract_tuner_trial_progress(warmup_tuner)
    else:
      logging.info('MetaLearning Algorithm not provided.')

    # Create new fn_args for final tuning stage.
    fn_args = fn_args_utils.get_common_fn_args(
        input_dict, exec_properties, working_dir=self._get_tmp_dir())
    tuner_fn = udf_utils.get_fn(exec_properties, 'tuner_fn')
    tuner_fn_result = tuner_fn(fn_args)
    tuner_fn_result.tuner.oracle.max_trials = max(
        (tuner_fn_result.tuner.oracle.max_trials - warmup_trials), 1)
    tuner = self.search(tuner_fn_result)
    tuner_trial_data = extract_tuner_trial_progress(tuner)

    if warmup_trial_data:
      cumulative_tuner_trial_data, best_tuner_ix = merge_trial_data(
          warmup_trial_data, tuner_trial_data)
      cumulative_tuner_trial_data['warmup_trial_data'] = warmup_trial_data[
          BEST_CUMULATIVE_SCORE]
      cumulative_tuner_trial_data['tuner_trial_data'] = tuner_trial_data[
          BEST_CUMULATIVE_SCORE]

      if isinstance(tuner.oracle.objective, kerastuner.Objective):
        cumulative_tuner_trial_data['objective'] = tuner.oracle.objective.name
      else:
        cumulative_tuner_trial_data['objective'] = 'objective not understood'

      tuner_trial_data = cumulative_tuner_trial_data
      best_tuner = warmup_tuner if best_tuner_ix == 0 else tuner
    else:
      best_tuner = tuner
    tfx_tuner.write_best_hyperparameters(best_tuner, output_dict)
    tuner_plot_path = os.path.join(
        artifact_utils.get_single_uri(output_dict['trial_summary_plot']),
        'tuner_plot_data.txt')
    io_utils.write_string_file(tuner_plot_path, json.dumps(tuner_trial_data))
    logging.info('Tuner plot data written at: %s', tuner_plot_path)
