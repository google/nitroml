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
from typing import Any, Dict, List, Type

from absl import logging
import kerastuner
from kerastuner.engine import base_tuner
from tfx import types
from tfx.components.base import base_executor
from tfx.components.trainer import fn_args_utils
from tfx.components.tuner import executor as tfx_tuner
from tfx.components.tuner.component import TunerFnResult
from tfx.components.util import udf_utils
from tfx.types import artifact_utils
from tfx.utils import io_utils

DEFAULT_WARMUP_TRIALS = 4
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
    """A Tuner which dyanamically inherits tuner_class and implements trial callbacks."""

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

    Raises:
      ValueError: When the tuner is not of type TrialTrackingTuner.
  """

  classname = tuner.__class__.__qualname__
  if classname == CUSTOM_TUNER_NAME:
    return tuner.get_tuner_plot_data()
  else:
    raise ValueError(
        f"Tuner is expected to have the class {CUSTOM_TUNER_NAME}, but got {classname}."
        "Use `get_tuner_cls_with_callbacks()` to define the kerastuner.")


def merge_trial_data(*all_tuner_data: Dict[str, Any]) -> Dict[str, Any]:
  """Merges sorted trial progress from multiple tuners of type TrialTrackingTuner.

    Args:
      all_tuner_data: List of tuner data stored as dictionary.

    Raises:
    ValueError: If the list `all_tuner_data` is None or empty.
  """

  if not all_tuner_data:
    raise ValueError('The arg `all_tuner_data` cannot be empty or None.')

  obj_direction = all_tuner_data[0][OBJECTIVE_DIRECTION]
  is_ascending = obj_direction == 'max'
  best_score_so_far = sys.float_info.min if is_ascending else sys.float_info.max
  best_performing_tuner = 0
  best_tuner_score = []
  cumulative_best_score = []

  for tuner_data in all_tuner_data:
    # sorted score list from the tuner.
    trial_data = tuner_data[BEST_CUMULATIVE_SCORE]
    best_tuner_score.append(trial_data[-1])

    if is_ascending:
      trial_data = [
          score if (score > best_score_so_far) else best_score_so_far
          for score in trial_data
      ]
    else:
      trial_data = [
          score if (score < best_score_so_far) else best_score_so_far
          for score in trial_data
      ]

    cumulative_best_score.extend(trial_data)
    best_score_so_far = cumulative_best_score[-1]

  cumulative_trial_data = {}
  cumulative_trial_data[BEST_CUMULATIVE_SCORE] = cumulative_best_score
  cumulative_trial_data[OBJECTIVE_DIRECTION] = obj_direction

  # Find the best tuner.
  if obj_direction == 'max':
    best_performing_tuner = max(enumerate(best_tuner_score), key=lambda x: x[1])
  else:
    best_performing_tuner = min(enumerate(best_tuner_score), key=lambda x: x[1])

  return (cumulative_trial_data, best_performing_tuner[0])


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

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:

    if tfx_tuner.get_tune_args(exec_properties):
      raise ValueError("TuneArgs is not supported by this Tuner's Executor.")

    tuner_fn = udf_utils.get_fn(exec_properties, 'tuner_fn')

    # Perform warmup tuning if WARMUP_HYPERPARAMETERS given.
    warmup_trials = 0
    warmup_trial_data = None
    if input_dict.get(WARMUP_HYPERPARAMETERS):
      hyperparameters_file = io_utils.get_only_uri_in_dir(
          artifact_utils.get_single_uri(input_dict[WARMUP_HYPERPARAMETERS]))
      hparams_warmup_config = json.loads(
          io_utils.read_string_file(hyperparameters_file))
      warmup_trials = DEFAULT_WARMUP_TRIALS
      fn_args = fn_args_utils.get_common_fn_args(
          input_dict,
          exec_properties,
          working_dir=self._get_tmp_dir() + 'warmup')
      fn_args.custom_config[WARMUP_HYPERPARAMETERS] = hparams_warmup_config
      fn_args.custom_config['max_trials'] = warmup_trials
      warmtuner_fn_result = tuner_fn(fn_args)
      warmup_tuner = self.search(warmtuner_fn_result)
      warmup_trial_data = extract_tuner_trial_progress(warmup_tuner)

    # Create new fn_args for final tuning stage.
    fn_args = fn_args_utils.get_common_fn_args(
        input_dict, exec_properties, working_dir=self._get_tmp_dir())
    tuner_fn_result = tuner_fn(fn_args)
    tuner_fn_result.tuner.oracle.max_trials = max(
        (tuner_fn_result.tuner.oracle.max_trials - warmup_trials), 1)
    tuner = self.search(tuner_fn_result,)
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
