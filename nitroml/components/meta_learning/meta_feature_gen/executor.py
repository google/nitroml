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
"""Executor for MetaFeatureGen."""

import os
import json
from typing import Any, Dict, List, Text

from absl import logging
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.components.base import base_executor
from tfx.types.artifact import Artifact
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import statistics_pb2

# Default file name for generated MetaFeature file.
_DEFAULT_FILE_NAME = 'meta_features.txt'


class MetaFeatureGenExecutor(base_executor.BaseExecutor):
  """Executor for MetaFeatureGen."""

  def Do(self, input_dict: Dict[Text, List[Artifact]],
         output_dict: Dict[Text, List[Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Generate MetaFeatures for meta training datasets.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - statistics: output from StatisticsGen component.
      output_dict: Output dict from key to a list of artifacts.
      exec_properties: A dict of execution properties
    """

    custom_config = exec_properties['custom_config']

    train_stats_uri = io_utils.get_only_uri_in_dir(
        artifact_utils.get_split_uri(input_dict['statistics'], 'train'))

    stats = tfdv.load_statistics(train_stats_uri)

    if len(stats.datasets) != 1:
      raise ValueError(
          'DatasetFeatureStatisticsList proto contains multiple datasets. Only '
          'one dataset is currently supported.')
    stats = stats.datasets[0]

    num_float_features = 0
    num_int_features = 0
    num_categorical_features = 0
    for feature in stats.features:
      name = feature.name
      if not name:
        name = feature.path.step[0]
      logging.info('Feature Name: %s', name)

      if statistics_pb2.FeatureNameStatistics.FLOAT == feature.type:
        num_float_features += 1
      elif statistics_pb2.FeatureNameStatistics.INT == feature.type:
        num_int_features += 1
      else:
        num_categorical_features += 1

    meta_feature_dict = {
        'num_examples': stats.num_examples,
        'num_int_features': num_int_features,
        'num_float_features': num_float_features,
        'num_categorical_features': num_categorical_features,
    }

    meta_feature_path = os.path.join(
        artifact_utils.get_single_uri(output_dict['meta_features']),
        _DEFAULT_FILE_NAME)

    io_utils.write_string_file(meta_feature_path, json.dumps(meta_feature_dict))
    logging.info('MetaFeature saved at %s', meta_feature_path)
