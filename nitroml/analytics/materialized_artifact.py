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
from typing import Dict, Type
import pandas as pd
from tensorflow.io import gfile
from tfx import types
from tfx.utils import abc_utils


class MaterializedArtifact(abc.ABC):
  """TFX artifact used for artifact analysis and visualization."""

  def __init__(self, artifact: types.Artifact):
    self._artifact = artifact

  def __str__(self):
    return 'Type: %s, URI: %s' % (self.type_name, self.uri)

  def __repr__(self):
    return f'<{self.__str__()}>'

  # Artifact type (of type `Type[types.Artifact]`).
  ARTIFACT_TYPE = abc_utils.abstract_property()

  @property
  def uri(self) -> str:
    """Artifact URI."""
    return self._artifact.uri

  @property
  def id(self) -> int:
    """Artifact id."""
    return self._artifact.id

  @property
  def name(self) -> str:
    """Artifact name."""
    return self._artifact.mlmd_artifact.name or self._artifact.name

  @property
  def type_name(self) -> str:
    """Artifact type name."""
    return self._artifact.type_name

  @property
  def producer_component(self) -> str:
    """The producer component of this artifact."""
    return self._artifact.producer_component

  @property
  def properties(self) -> Dict[str, str]:
    """Returns dictionary of custom and default properties of the artifact."""
    properties = {}
    for key, value in self._artifact.mlmd_artifact.properties.items():
      properties[key] = value.string_value

    for key, value in self._artifact.mlmd_artifact.custom_properties.items():
      properties[key] = value.string_value

    return properties

  def _validate_payload(self):
    """Raises error if the artifact uri is not readable.

    Raises:
      IOError: Error raised if no conclusive determination could be made
      of files state (either because the path definitely does not exist or
      because some error occurred trying to query the file's state).
    """

    if not gfile.exists(self.uri):
      raise IOError(f'Artifact URI {self.uri} not readable.')

  @abc.abstractmethod
  def show(self) -> None:
    """Displays respective visualization for artifact type."""
    raise NotImplementedError()


class GenericMaterializedArtifact(MaterializedArtifact):
  """A Generic Artifact class to assign unregistered artifact types."""

  ARTIFACT_TYPE = types.Artifact

  def show(self) -> None:
    """Displays respective visualization for artifact type."""
    raise NotImplementedError("Artifact type '%s' not registered." %
                              self.type_name)


class ArtifactRegistry:
  """Registry of artifact definitions."""

  def __init__(self):
    self.artifacts = {}

  def __repr__(self) -> str:
    return repr(self.artifacts)

  def _repr_html_(self) -> str:
    return pd.DataFrame.from_dict(self.artifacts, orient='index',
                                  columns=['Artifact']).to_html()

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
    return self.artifacts.get(artifact_type_name, GenericMaterializedArtifact)


_REGISTRY = ArtifactRegistry()


def get_registry():
  return _REGISTRY
