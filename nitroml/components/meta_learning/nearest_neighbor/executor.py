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
"""Executor for NearestNeighborMetaLearner."""

import json
from typing import Any, Dict, List, Text

from absl import logging
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils import io_utils
import tensorflow_data_validation as tfdv

_MAX_INPUTS = 10

# For metafeatures, we need common attributes:

# Examples: Number of int features?

class NearestNeighborMetaLearnerExecutor(base_executor.BaseExecutor):
  """Executor for NearestNeighborMetaLearnerExecutor."""

  def Do(self, input_dict: Dict[Text, List[Artifact]],
         output_dict: Dict[Text, List[Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Recommends a tuner config using Nearest Neighbor w.r.t MetaFeatures.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - meta_train_statistics: list of statistics of train datasets
        - meta_test_statistics: list of statistics of test datasets
      output_dict: Output dict from key to a list of artifacts, currently unused.
      exec_properties: A dict of execution properties
    """
    custom_config = exec_properties['custom_config']
    custom_config =
    train_stats = {}
    for ix in range(_MAX_INPUTS):
      stats_key = f'train_statistics_{ix}'
      if (stats_key in input_dict):
        train_stats_uri = io_utils.get_only_uri_in_dir(
            artifact_utils.get_split_uri(input_dict[stats_key], 'train'))

        stats = tfdv.load_statistics(train_stats_uri)
        if len(stats.datasets) != 1:
          raise ValueError(
              'DatasetFeatureStatisticsList proto contains multiple datasets. Only '
              'one dataset is currently supported.')

        stats = stats.datasets[0]

        for feature in stats.features:

        logging.info('Number of examples: {%s}', stats.num_examples)
        logging.info(stats.features)
