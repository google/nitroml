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
"""A collection of objects to analyze and visualize Machine Learning Metadata."""

from typing import Any, Dict, ItemsView, KeysView, Optional, Type, Union, ValuesView

from nitroml.analytics import materialized_artifact
from nitroml.analytics import standard_materialized_artifacts
import pandas as pd
from tfx import types

from ml_metadata.google.services.mlmd_service.proto import mlmd_service_pb2
from ml_metadata.google.tfx import metadata_store
from ml_metadata.proto import metadata_store_pb2


def register_standard_artifacts():
  """Registers all artifact classes in 'standard_materialized_artifacts.py'."""
  standard_materialized_artifacts.register_standard_artifacts()


def register_artifact_class(artifact_class:
                            Type[materialized_artifact.MaterializedArtifact]):
  """Registers a subclass of MaterializedArtifact to artifact registery.

  If an artifact_class is registered with the same ARTIFACT_TYPE field,
  it is replaced.

  Args:
    artifact_class: A subclass of MaterializedArtifact to be registered.
  """
  if not issubclass(artifact_class, materialized_artifact.MaterializedArtifact):
    raise ValueError("Artifact class does not subclass "
                     "'materialized_artifact.MaterializedArtifact'")
  materialized_artifact.get_registry().register(artifact_class)


class PropertyDictWrapper:
  """Wraps a dictionary object for property access functionality.

  This class is read-only (setting properties is not implemented).
  """

  def __init__(self, data: Dict[str, Any]):
    self._data = data.copy()

  def __getitem__(self, key):
    return self._data[key]

  def __getattr__(self, key):
    try:
      return self._data[key]
    except KeyError:
      raise AttributeError

  def __contains__(self, val):
    return val in self._data

  def __repr__(self):
    return repr(self._data)

  def get_all(self) -> Dict[str, Any]:
    """Returns dictionary representation of this object."""
    return self._data.copy()

  def keys(self) -> KeysView[str]:
    """Returns list of keys in this object."""
    return self._data.keys()

  def values(self) -> ValuesView[Any]:
    """Returns list of values in this object."""
    return self._data.values()

  def items(self) -> ItemsView[str, Any]:
    """Returns list of key, value pairs in this object."""
    return self._data.items()


class ComponentRun:
  """An object representation of MLMD Component.

  Attributes:
    run_id: The id of the pipeline run that created this component.
  """

  def __init__(self, run_id: str, execution: metadata_store_pb2.Execution,
               store: metadata_store.MetadataStore):
    """Initializes instance of ComponentRun in a given pipeline run.


    Args:
      run_id: The id of the pipeline run that created this component.
      execution: Execution object containing component execution properties.
      store: A store for the artifact metadata.
    """
    self.run_id = run_id
    self._execution = execution
    self._store = store

  @property
  def component_name(self):
    """The name of this component."""
    return self._execution.property['component_id'].string_value

  @property
  def inputs(self) -> PropertyDictWrapper:
    """List of input artifacts in this components workflow.

    Raises:
      NotImplementedError: If this functionality has yet to be implemented.
    """
    raise NotImplementedError(
        'ComponentRun.inputs() is currently unimplemented.')

  @property
  def outputs(self) -> PropertyDictWrapper:
    """List of output artifacts in this components workflow."""
    [component_run_context] = [
        context
        for context in self._store.get_contexts_by_execution(self._execution.id)
        if 'component_id' in context.properties
    ]
    artifacts = self._store.get_artifacts_by_context(component_run_context.id)
    output = {}
    artifact_types = self._store.get_artifact_types_by_id(
        [artifact.type_id for artifact in artifacts])
    for artifact, artifact_type in zip(artifacts, artifact_types):
      tfx_artifact = types.Artifact(artifact_type)
      tfx_artifact.set_mlmd_artifact(artifact)
      materialized_artifact_class = materialized_artifact.get_registry(
      ).get_artifact_class(tfx_artifact.type_name)
      output[tfx_artifact.name] = materialized_artifact_class(tfx_artifact)

    return PropertyDictWrapper(output)

  @property
  def exec_properties(self) -> Dict[str, Any]:
    """A dictionary of user defined exec properties of this component."""
    return self._execution.properties

  @property
  def describe(self) -> pd.Series:
    """Table of system defined execution detail in this component.

    Raises:
      NotImplementedError: If this functionality has yet to be implemented.
    """
    raise NotImplementedError(
        'ComponentRun.describe() is currently unimplemented.')


class PipelineRun:
  """A class representing a single pipeline run.

  Attributes:
    name: The human-readable name of the pipeline.
    run_id: The unique id of the pipeline.
  """

  def __init__(self, name: str, run_id: str, context_id: int,
               store: metadata_store.MetadataStore):
    """Initializes an instance of PipelineRun with an existing run id.

    Args:
      name: The human-readable name of the pipeline.
      run_id: The unique id of the pipeline.
      context_id: The id of the respective context object to this run.
      store: A store for the artifact metadata.
    """
    self.name = name
    self.run_id = run_id
    self._context_id = context_id
    self._store = store

  @property
  def components(self) -> Dict[str, ComponentRun]:
    """A dictionary of (Component Name, ComponentRun) key-value pairs."""
    components = {}
    for execution in self._store.get_executions_by_context(self._context_id):
      component_name = execution.properties['component_id'].string_value
      components[component_name] = ComponentRun(self.run_id, execution,
                                                self._store)

    return components

  def show(self):
    """Displays a DAG visualization of this run and its components."""
    raise NotImplementedError('PipelineRun.show() is currently unimplemented.')


class Analytics:
  """Querying tool to analyze and visualize your machine learning metadata."""

  def __init__(self,
               config: Union[metadata_store_pb2.ConnectionConfig,
                             metadata_store_pb2.MetadataStoreClientConfig,
                             mlmd_service_pb2.MLMDServiceClientConfig] = None,
               store: Optional[metadata_store.MetadataStore] = None):
    """Connects instance of analytics class to the given connection configuration.

    Note: While store parameter is optional, either 'config' or 'store' must be
    provided to successfully initialize an 'Analytics' instance.

    Args:
      config: Configuration to connect to the database or the metadata store
        server.
      store: A store for the artifact metadata.
    """
    if bool(store) == bool(config):
      raise ValueError('Expected exactly one of store or config')
    self._store = store if store else metadata_store.MetadataStore(config)

  def _get_runs(self) -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary of runs with pipeline names, runids, and context ids."""
    ctxs = self._store.get_contexts_by_type('run')
    runs = {}
    for ctx in ctxs:
      run_id = ctx.properties['run_id'].string_value
      runs[run_id] = {
          'pipeline_name': ctx.properties['pipeline_name'].string_value,
          'run_id': run_id,
          'context_id': ctx.id
      }
    return runs

  def list_runs(self) -> pd.DataFrame:
    """Returns a dataframe of pipeline names and runids."""
    return pd.DataFrame(self._get_runs().values())[['pipeline_name', 'run_id']]

  def get_run(self, run_id: str) -> PipelineRun:
    """Returns a Run object with the given run_id.

    Args:
      run_id: The unique id of the pipeline to query.

    Raises:
      KeyError: If run_id does not exist in the metadata.
    """
    runs = self._get_runs()
    if run_id not in runs:
      raise ValueError('Run ID "%s" not found in metadata store.' % run_id)
    return PipelineRun(runs[run_id]['pipeline_name'], run_id,
                       runs[run_id]['context_id'], self._store)
