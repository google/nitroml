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

import collections
from typing import Type

import pandas as pd
from tfx import types

from google3.pyglib import gfile


class MaterializedArtifact:
  """TFX artifact used for artifact analysis and visualization."""

  def __init__(self, artifact: types.Artifact):
    self.artifact = artifact

  def __str__(self):
    return f'{self.artifact.artifact_type} Artifact'

  def __repr__(self):
    return f'<{self.__str__()}>'

  # Artifact type (of type `Type[types.Artifact]`).
  ARTIFACT_TYPE = types.Artifact

  def _validate_payload(self):
    """Raises error if the artifact uri is not readable.

    Raises:
      IOError: Error raised if no conclusive determination could be made
      of files state (either because the path definitely does not exist or
      because some error occurred trying to query the file's state).
    """

    if not gfile.Readable(self.artifact.uri):
      raise IOError(f'Artifact URI {self.artifact.uri} not readable.')

  def show(self) -> None:
    """Displays respective visualization for artifact type."""
    raise NotImplementedError("Artifact type '%s' not registered." %
                              self.artifact.type_name)

  def to_dataframe(self) -> pd.DataFrame:
    """Returns dataframe representation of the artifact."""
    properties = collections.defaultdict(list)
    for key, value in self.artifact.mlmd_artifact.properties.items():
      properties['Property'].append(key)
      properties['Value'].append(value.string_value)

    for key, value in self.artifact.mlmd_artifact.custom_properties.items():
      properties['Property'].append(key)
      properties['Value'].append(value.string_value)

    return pd.DataFrame(properties)


class ArtifactRegistry(object):
  """Registry of artifact definitions."""

  def __init__(self):
    self.artifacts = {}

  def register(self, artifact_class: Type[MaterializedArtifact]):
    artifact_type = artifact_class.ARTIFACT_TYPE
    if not (issubclass(artifact_type, types.Artifact) and
            artifact_type.TYPE_NAME is not None):
      raise TypeError(
          'Artifact class must provide subclass of types.Artifact in its '
          'ARTIFACT_TYPE attribute. This subclass must have non-None TYPE_NAME '
          'attribute.')
    self.artifacts[artifact_type.TYPE_NAME] = artifact_class

  def get_artifact_class(self,
                         artifact_type_name: str) -> Type[MaterializedArtifact]:
    return self.artifacts.get(artifact_type_name, MaterializedArtifact)


_REGISTRY = ArtifactRegistry()


def get_registry():
  return _REGISTRY
