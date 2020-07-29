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
"""Executor for MetaLearner."""

import collections
import json
import os
from typing import Any, Dict, List

from absl import logging
import kerastuner
from nitroml.components.metalearning import artifacts
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils import io_utils

_DEFAULT_FILE_NAME = 'meta_hyperparameters.txt'

MAX_INPUTS = 10
OUTPUT_MODEL = 'metalearned_model'
OUTPUT_HYPERPARAMS = 'output_hyperparameters'
MAJORITY_VOTING = 'majority_voting'
METALEARNING_ALGORITHMS = [
    MAJORITY_VOTING,
]


class MetaLearnerExecutor(base_executor.BaseExecutor):
  """Executor for MetaLearnerExecutor."""

  def _create_search_space_using_voting(
      self, candidate_hparams: List[Dict[str,
                                         Any]]) -> kerastuner.HyperParameters:
    """Convert List of HParams to kerastuner.HyperaParameters.

    Args:
      candidate_hparams: List of Dict of HParams with same keys.

    Returns:
      discrete_search_space: A kerastuner.HyperParameters object representing
      a discrete search
      space created using voting. For example, when

      `candidate_hparams` is [{`learning_rate`: 0.01, `num_nodes`: 32},
                              {`learning_rate`: 0.001, `num_nodes`: 128},
                              {`learning_rate`: 0.01, `num_nodes`: 128},
                              {`learning_rate`: 0.001, `num_nodes`: 128}]

      then, `discrete_search_space` depicts the following discrete search
      space
      {`learning_rate`: [0.01, 0.001], `num_nodes`: [128]}

    Raises:
      ValueError: An error occured when `candidate_hparams` is empty or None.
    """

    if not candidate_hparams:
      raise ValueError(
          f'Expected a non-empty list of candidate_hparams. Got {candidate_hparams}'
      )

    hparams = candidate_hparams[0].keys()

    search_space = {}
    for key in hparams:
      search_space[key] = collections.Counter(
          [candidate[key] for candidate in candidate_hparams]).most_common()

    discrete_search_space = kerastuner.HyperParameters()

    for key, value_list in search_space.items():
      max_vote = -1
      candidate_values = []
      for value, count in value_list:
        if count >= max_vote:
          candidate_values.append(value)
        else:
          break

      discrete_search_space.Choice(
          key, candidate_values, default=candidate_values[0])

    return discrete_search_space

  def Do(self, input_dict: Dict[str, List[Artifact]],
         output_dict: Dict[str, List[Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Recommends a tuner config.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - meta_train_features_N: MetaFeatures for Nth train dataset.
        - hparams_train_N: HParms for Nth train dataset. The maximum value `N`
          being _MAX_INPUTS.
      output_dict: Output dict from key to a list of artifacts.
      exec_properties: A dict of execution properties.

    Raises:
    """

    algorithm = exec_properties['algorithm']
    custom_config = exec_properties['custom_config']

    train_stats = {}
    # This should be agnostic to meta-feature type.
    for ix in range(MAX_INPUTS):
      metafeature_key = f'meta_train_features_{ix}'
      if metafeature_key in input_dict:
        metafeature_uri = os.path.join(
            artifact_utils.get_single_uri(input_dict[metafeature_key]),
            artifacts.MetaFeatures.DEFAULT_FILE_NAME)
        logging.info('Found %s at %s.', metafeature_key, metafeature_uri)
        metafeatures = json.loads(io_utils.read_string_file(metafeature_uri))
        # Only logging metafeatures for now. MetaFeature will be used in
        # upcoming algorithms.
        logging.info('metafeatures %s.', metafeatures['metafeature'])

    all_hparams = []
    for ix in range(MAX_INPUTS):
      hparam_key = f'hparams_train_{ix}'
      if hparam_key in input_dict:
        hyperparameters_file = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(input_dict[hparam_key]))
        logging.info('Found %s at %s.', hparam_key, hyperparameters_file)
        hparams_json = json.loads(
            io_utils.read_string_file(hyperparameters_file))
        all_hparams.append(hparams_json['values'])
        logging.info('File %s.', hparams_json)

    if algorithm == 'majority_voting':

      discrete_search_space = self._create_search_space_using_voting(
          all_hparams)
      voted_hparams_config = discrete_search_space.get_config()
      logging.info('Discrete Search Space: %s', voted_hparams_config)

      meta_hparams_path = os.path.join(
          artifact_utils.get_single_uri(output_dict[OUTPUT_HYPERPARAMS]),
          _DEFAULT_FILE_NAME)

      io_utils.write_string_file(meta_hparams_path,
                                 json.dumps(voted_hparams_config))
      logging.info('Meta HParams saved at %s', meta_hparams_path)

    else:
      raise NotImplementedError(
          f'The algorithm "{algorithm}" is not supported.')
