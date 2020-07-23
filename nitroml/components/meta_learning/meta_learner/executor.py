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

import os
import json
from typing import Any, Dict, List, Text

from nitroml.components.meta_learning import artifacts
from absl import logging
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils import io_utils
import tensorflow_data_validation as tfdv

_MAX_INPUTS = 10


class MetaLearnerExecutor(base_executor.BaseExecutor):
  """Executor for MetaLearnerExecutor."""

  def Do(self, input_dict: Dict[Text, List[Artifact]],
         output_dict: Dict[Text, List[Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Recommends a tuner config.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - meta_train_statistics: list of statistics of train datasets
        - meta_test_statistics: list of statistics of test datasets
      output_dict: Output dict from key to a list of artifacts, currently unused.
      exec_properties: A dict of execution properties
    """

    algorithm = exec_properties['algorithm']
    custom_config = exec_properties['custom_config']

    train_stats = {}
    for ix in range(_MAX_INPUTS):
      meta_feature_key = f'meta_train_features_{ix}'
      if (meta_feature_key in input_dict):
        meta_feature_uri = os.path.join(
            artifact_utils.get_single_uri(input_dict[meta_feature_key]),
            artifacts.MetaFeatures.DEFAULT_FILE_NAME)
        logging.info('Found %s at %s.', meta_feature_key, meta_feature_uri)

        meta_features = jsonio_utils.read_string_file(meta_feature_uri)
        logging.info('File %s.', data)
      else:
        logging.info('Did not Find %s.', meta_feature_key)