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

import json
import os
from typing import Any, Dict, List, Text

from absl import logging
from nitroml.automl.metalearning import artifacts
import tensorflow_data_validation as tfdv
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils import io_utils

from tensorflow_metadata.proto.v0 import statistics_pb2

EXAMPLES_KEY = 'transformed_examples'
STATISTICS_KEY = 'statistics'
METAFEATURES_KEY = 'metafeatures'


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

    train_stats_uri = io_utils.get_only_uri_in_dir(
        artifact_utils.get_split_uri(input_dict[STATISTICS_KEY], 'train'))

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

      # For structured fields, name is set by path and is not in the name
      # attribute.
      if not name:
        name = feature.path.step[0]
      logging.info('Feature Name: %s', name)

      if statistics_pb2.FeatureNameStatistics.FLOAT == feature.type:
        num_float_features += 1
      elif statistics_pb2.FeatureNameStatistics.INT == feature.type:
        num_int_features += 1
      else:
        num_categorical_features += 1

    metafeature_dict = {
        'num_examples': stats.num_examples,
        'num_int_features': num_int_features,
        'num_float_features': num_float_features,
        'num_categorical_features': num_categorical_features,
    }

    metafeature_dict['metafeature'] = [
        stats.num_examples, num_int_features, num_float_features,
        num_categorical_features
    ]

    metafeature_path = os.path.join(
        artifact_utils.get_single_uri(output_dict[METAFEATURES_KEY]),
        artifacts.MetaFeatures.DEFAULT_FILE_NAME)

    io_utils.write_string_file(metafeature_path, json.dumps(metafeature_dict))
    logging.info('MetaFeature saved at %s', metafeature_path)
