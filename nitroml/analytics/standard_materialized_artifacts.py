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

from nitroml.analytics import materialized_artifact
from tfx.orchestration.experimental.interactive import standard_visualizations
from tfx.types import standard_artifacts


class StatisticsGenArtifact(materialized_artifact.MaterializedArtifact):
  """Visualization for standard_artifacts.Statistics."""

  ARTIFACT_TYPE = standard_artifacts.ExampleStatistics

  def show(self):
    self._validate_payload()
    standard_visualizations.ExampleStatisticsVisualization().display(
        self.artifact)


class SchemaGenArtifact(materialized_artifact.MaterializedArtifact):
  """Visualization for standard_artifacts.Schema."""

  ARTIFACT_TYPE = standard_artifacts.Schema

  def show(self):
    standard_visualizations.SchemaVisualization().display(self.artifact)

_STANDARD_ARTIFACTS = frozenset([StatisticsGenArtifact, SchemaGenArtifact])


def register_standard_artifacts():
  for artifact in _STANDARD_ARTIFACTS:
    materialized_artifact.get_registry().register(artifact)
