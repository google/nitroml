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

from inspect import ismethod
import json
import os
from typing import Any, Dict, List, Type

from absl import logging
from google.protobuf import json_format
from kerastuner.engine import base_tuner
from tfx import types
from tfx.components.base import base_executor
from tfx.components.trainer import fn_args_utils
from tfx.components.tuner import executor as tfx_tuner
from tfx.components.util import udf_utils
from tfx.components.trainer import fn_args_utils
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils


def get_tuner_cls_with_callbacks(tuner_class: Type[base_tuner.BaseTuner]):
  """Returns a CustomTuner class which overrides the tuner trial callbacks.

    Args:
      tuner_class: An existing tuner class that extends the base_tuner.BaseTuner.
  """

  class CustomTuner(tuner_class):  # pylint: disable=E0239

    def on_search_begin(self):
      super(CustomTuner, self).on_search_begin()
      self._trial_plot_data = {}
      self._trial_plot_data['best_trial_score'] = []

    def on_trial_end(self, trial):
      super(CustomTuner, self).on_trial_end(trial)
      best_trial = self.oracle.get_best_trials()[0]
      self._trial_plot_data['best_trial_score'].append(best_trial.score)

    def get_tuner_plot_data(self):
      return self._trial_plot_data

  return CustomTuner


class Executor(base_executor.BaseExecutor):
  """The executor for nitroml.components.tuner.components.Tuner."""

  def search(self, input_dict: Dict[str, List[types.Artifact]],
             exec_properties: Dict[str, Any],
             working_dir: str) -> base_tuner.BaseTuner:
    """Conduct a single hyperparameter search loop, and return the Tuner."""

    tuner_fn = udf_utils.get_fn(exec_properties, 'tuner_fn')
    fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties,
                                               working_dir)

    tuner_fn_result = tuner_fn(fn_args)
    tuner = tuner_fn_result.tuner

    tuner.search_space_summary()
    logging.info('Start tuning... Tuner ID: %s', tuner.tuner_id)
    tuner.search(**tuner_fn_result.fit_kwargs)
    logging.info('Finished tuning... Tuner ID: %s', tuner.tuner_id)
    tuner.results_summary()

    return tuner

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:

    if tfx_tuner.get_tune_args(exec_properties):
      raise ValueError("TuneArgs is not supported by this Tuner's Executor.")

    tuner = self.search(input_dict, exec_properties, self._get_tmp_dir())
    tfx_tuner.write_best_hyperparameters(tuner, output_dict)

    if hasattr(tuner, 'get_tuner_plot_data') and ismethod(
        getattr(tuner, 'get_tuner_plot_data')):

      tuner_plot_path = os.path.join(
          artifact_utils.get_single_uri(output_dict['trial_summary_plot']),
          'tuner_plot_data.txt')
      io_utils.write_string_file(tuner_plot_path,
                                 json.dumps(tuner.get_tuner_plot_data()))
      logging.info('Tuner plot data written at: %s', tuner_plot_path)
