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
"""A generic Materialized Artifact definition."""

import abc
import pandas as pd

from tfx.orchestration.experimental.interactive import visualizations


class MaterializedArtifact(visualizations.ArtifactVisualization):
  """Abstract base class for materialized artifacts.

  Represents an output of a tfx component that has been materialized on disk.
  Subclasses provide implementations to load and display a specific artifact
  type.
  """

  @abc.abstractmethod
  def to_dataframe(self) -> pd.DataFrame:
    """Returns dataframe representation of the artifact."""
