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
from nitroml.automl.metalearning import artifacts
import numpy as np
import tensorflow as tf
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils import io_utils
from tfx.utils import path_utils

_DEFAULT_FILE_NAME = 'meta_hyperparameters.txt'

MAX_INPUTS = 10
OUTPUT_MODEL = 'metamodel'
OUTPUT_HYPERPARAMS = 'output_hyperparameters'
MAJORITY_VOTING = 'majority_voting'
NEAREST_NEIGHBOR = 'nearest_neighbor'
METALEARNING_ALGORITHMS = [
    MAJORITY_VOTING,
    NEAREST_NEIGHBOR,
]


class MetaLearnerExecutor(base_executor.BaseExecutor):
  """Executor for MetaLearnerExecutor."""

  def _convert_to_kerastuner_hyperparameters(
      self,
      candidate_hparams: List[Dict[str,
                                   Any]]) -> List[kerastuner.HyperParameters]:
    """Convert list of HSpace to a list of search space each with cardinality 1.

    Args:
      candidate_hparams: List of Dict of HParams with same keys.

    Returns:
      The list of hparams in the search space.
    """

    if not candidate_hparams:
      raise ValueError(
          f'Expected a non-empty list of candidate_hparams. Got {candidate_hparams}'
      )

    simple_search_space_list = []
    for candidate_hparam in candidate_hparams:
      simple_search_space = kerastuner.HyperParameters()
      for key in candidate_hparam:
        simple_search_space.Choice(key, [candidate_hparam[key]])
      simple_search_space_list.append(simple_search_space)
    return simple_search_space_list

  def _create_search_space_using_voting(
      self, candidate_hparams: List[Dict[str,
                                         Any]]) -> kerastuner.HyperParameters:
    """Convert List of HParams to kerastuner.HyperaParameters based on voting.

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
          max_vote = count
        else:
          break

      discrete_search_space.Choice(
          key, candidate_values, default=candidate_values[0])

    return discrete_search_space

  def _create_knn_model_from_metafeatures(
      self, metafeatures_list: List[List[float]]) -> tf.keras.Model:
    """Creates a Model that stores metafeatures as a Layer for nearest neighbor.

    The function creates a keras model with a dense layer. The weight kernel of
    the layer is formed of metafeatures of training datasets. One can find the
    nearest neighbor for a new dataset by doing a forward pass of the model. The
    output of the forward pass represents the similarity scores based on the
    inner product of metafeatures. The keras model is intended to be used in
    metalearning initialized tuner, which receives the metafeatures of a new
    dataset.

    Args:
      metafeatures_list: List of metafeatures of training datasets.

    Returns:
      A tf.keras.Model with a single dense layer having metafeatures as weights.
    """

    n = len(metafeatures_list[0])
    k = len(metafeatures_list)
    inputs = tf.keras.layers.Input(shape=(n,))
    outputs = tf.keras.layers.Dense(
        k, activation=None, use_bias=False, name='metafeatures')(
            inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    weights = np.array(metafeatures_list, dtype=np.float32).T
    # Normalize weights to lie in a unit ball.
    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    model.get_layer('metafeatures').set_weights([weights])
    return model

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
    metafeatures_list = []
    # This should be agnostic to meta-feature type.
    for ix in range(MAX_INPUTS):
      metafeature_key = f'meta_train_features_{ix}'
      if metafeature_key in input_dict:
        metafeature_uri = os.path.join(
            artifact_utils.get_single_uri(input_dict[metafeature_key]),
            artifacts.MetaFeatures.DEFAULT_FILE_NAME)
        logging.info('Found %s at %s.', metafeature_key, metafeature_uri)
        metafeatures = json.loads(io_utils.read_string_file(metafeature_uri))
        metafeatures_list.append(metafeatures['metafeature'])

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

    if algorithm == MAJORITY_VOTING:
      discrete_search_space = self._create_search_space_using_voting(
          all_hparams)
      hparams_config_list = [discrete_search_space.get_config()]
    elif algorithm == NEAREST_NEIGHBOR:
      # Build nearest_neighbor model
      output_path = artifact_utils.get_single_uri(output_dict[OUTPUT_MODEL])
      serving_model_dir = path_utils.serving_model_dir(output_path)
      model = self._create_knn_model_from_metafeatures(metafeatures_list)
      # TODO(nikhilmehta): Consider adding signature here.
      model.save(serving_model_dir)

      # Collect all Candidate HParams
      hparams_list = self._convert_to_kerastuner_hyperparameters(all_hparams)
      hparams_config_list = [hparam.get_config() for hparam in hparams_list]
    else:
      raise NotImplementedError(
          f'The algorithm "{algorithm}" is not supported.')

    meta_hparams_path = os.path.join(
        artifact_utils.get_single_uri(output_dict[OUTPUT_HYPERPARAMS]),
        _DEFAULT_FILE_NAME)
    io_utils.write_string_file(meta_hparams_path,
                               json.dumps(hparams_config_list))
    logging.info('Meta HParams saved at %s', meta_hparams_path)
