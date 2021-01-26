# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX notebook visualizations for standard TFX artifacts."""

import os

from typing import List
from nitroml.analytics import materialized_artifact
import pandas as pd
import tensorflow as tf
from tensorflow.io import gfile
from tfx.orchestration.experimental.interactive import standard_visualizations
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class StatisticsGenArtifact(materialized_artifact.MaterializedArtifact):
  """Visualization for standard_artifacts.Statistics."""

  ARTIFACT_TYPE = standard_artifacts.ExampleStatistics

  def show(self):
    self._validate_payload()
    standard_visualizations.ExampleStatisticsVisualization().display(
        self._artifact)


class SchemaGenArtifact(materialized_artifact.MaterializedArtifact):
  """Visualization for standard_artifacts.Schema."""

  ARTIFACT_TYPE = standard_artifacts.Schema

  def show(self):
    standard_visualizations.SchemaVisualization().display(self._artifact)


class ExampleArtifact(materialized_artifact.MaterializedArtifact):
  """Visualization for standard_artifacts.Examples."""

  ARTIFACT_TYPE = standard_artifacts.Examples

  @property
  def split_names(self) -> List[str]:
    """A list of split names for the example data."""
    return artifact_utils.decode_split_names(self._artifact.split_names)

  def _load_row(self, ex):
    row = {}
    for k, v in ex.features.feature.items():
      if v.HasField('float_list'):
        if len(v.float_list.value) == 1:
          [row[k]] = v.float_list.value
      elif v.HasField('int64_list'):
        if len(v.int64_list.value) == 1:
          [row[k]] = v.int64_list.value
    return row

  def _load_table(self, ds):
    rows = []
    for record in ds.as_numpy_iterator():
      ex = tf.train.Example.FromString(record)
      rows.append(self._load_row(ex))

    return pd.DataFrame(rows)

  def to_dataframe(self, split: str, max_rows: int = 100) -> pd.DataFrame:
    """Returns dataframe representation of the artifact.

    Args:
      split: The name of the datasplit to be returned.
      max_rows: The maximum number of rows to be returned. If None, all rows in
        the split will be returned.
    """

    self._validate_payload()

    if max_rows and max_rows < 0:
      raise ValueError('`max_rows` must not be negative. Got: %d' % max_rows)
    filepaths = gfile.glob(os.path.join(self.uri, split, '*'))
    ds = tf.data.TFRecordDataset(filepaths, compression_type='GZIP').take(
        max_rows)
    return self._load_table(ds)

  def show(self):
    from IPython.core.display import display  # pylint: disable=g-import-not-at-top
    from IPython.core.display import HTML  # pylint: disable=g-import-not-at-top
    for split in self.split_names:
      display(HTML('<div><b>%r split:</b></div><br/>' % split))
      display(self.to_dataframe(split))


class ExampleAnomaliesArtifact(materialized_artifact.MaterializedArtifact):
  """Visualization for standard_artifacts.Statistics."""

  ARTIFACT_TYPE = standard_artifacts.ExampleAnomalies

  def show(self):
    self._validate_payload()
    standard_visualizations.ExampleAnomaliesVisualization().display(
        self._artifact)


class ModelEvaluationArtifact(materialized_artifact.MaterializedArtifact):

  ARTIFACT_TYPE = standard_artifacts.ModelEvaluation

  def show(self):
    self._validate_payload()
    standard_visualizations.ModelEvaluationVisualization().display(
        self._artifact)


_STANDARD_ARTIFACTS = frozenset(
    [StatisticsGenArtifact,
     SchemaGenArtifact,
     ExampleArtifact,
     ExampleAnomaliesArtifact,
     ModelEvaluationArtifact])


def register_standard_artifacts():
  for artifact in _STANDARD_ARTIFACTS:
    materialized_artifact.get_registry().register(artifact)
