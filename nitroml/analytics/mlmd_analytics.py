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

from typing import Any, Dict, ItemsView, KeysView, List, Optional, Set, Type, Union, ValuesView

from nitroml.analytics import materialized_artifact
from nitroml.analytics import standard_materialized_artifacts
import pandas as pd
from tfx import types

from ml_metadata.google.services.mlmd_service.proto import mlmd_service_pb2
from ml_metadata.google.tfx import metadata_store
from ml_metadata.proto import metadata_store_pb2

standard_materialized_artifacts.register_standard_artifacts()

# Type names for IR and Non-IR based MLMD instances respectively.
_NONIR_RUN_CONTEXT_NAME = 'run'
_IR_RUN_CONTEXT_NAME = 'pipeline_run'
_NONIR_COMPONENT_NAME = 'component_run'
_IR_COMPONENT_NAME = 'node'

_INPUT_EVENT_TYPES = frozenset({metadata_store_pb2.Event.DECLARED_INPUT,
                                metadata_store_pb2.Event.INPUT,
                                metadata_store_pb2.Event.INTERNAL_INPUT})
_OUTPUT_EVENT_TYPES = frozenset({metadata_store_pb2.Event.DECLARED_OUTPUT,
                                 metadata_store_pb2.Event.OUTPUT,
                                 metadata_store_pb2.Event.INTERNAL_OUTPUT})


def register_artifact_class(
    artifact_class: Type[materialized_artifact.MaterializedArtifact]):
  """Registers a subclass of MaterializedArtifact to artifact registery.

  If an artifact_class is registered with the same ARTIFACT_TYPE field,
  it is replaced.

  Args:
    artifact_class: A subclass of MaterializedArtifact to be registered.
  """
  if not issubclass(artifact_class, materialized_artifact.MaterializedArtifact):
    raise ValueError('Artifact class does not subclass '
                     '`materialized_artifact.MaterializedArtifact`')
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
               store: metadata_store.MetadataStore,
               context: metadata_store_pb2.Context):
    """Initializes instance of ComponentRun in a given pipeline run.


    Args:
      run_id: The id of the pipeline run that created this component.
      execution: Execution proto containing component execution properties.
      store: A store for the artifact metadata.
      context: Context proto to query artifacts.
    """
    self.run_id = run_id
    self._execution = execution
    self._store = store
    self._context = context

  @property
  def component_name(self):
    """The name of this component."""
    return self._context.name

  def _get_artifacts(
      self, event_type: Set['metadata_store_pb2.Event.Type']
  ) -> Dict[str, materialized_artifact.MaterializedArtifact]:
    """Returns artifacts associated with this component.

    Args:
      event_type: A set of Event Types to filter for.

    Returns:
      A dict of name, MaterializedArtifact pairs.
    """

    artifact_ids = [
        event.artifact_id for event in self._store.get_events_by_execution_ids(
            [self._execution.id]) if event.type in event_type
    ]
    artifacts = self._store.get_artifacts_by_id(artifact_ids)

    artifact_dict = {}
    artifact_types = self._store.get_artifact_types_by_id(
        [artifact.type_id for artifact in artifacts])
    for artifact, artifact_type in zip(artifacts, artifact_types):
      tfx_artifact = types.Artifact(artifact_type)
      tfx_artifact.set_mlmd_artifact(artifact)
      materialized_artifact_class = materialized_artifact.get_registry(
      ).get_artifact_class(tfx_artifact.type_name)
      artifact_dict[tfx_artifact.name] = materialized_artifact_class(
          tfx_artifact)  # pytype: disable=not-instantiable
    return artifact_dict

  @property
  def inputs(self) -> PropertyDictWrapper:
    """Dictionary of input artifacts in this component run."""
    return PropertyDictWrapper(
        self._get_artifacts(_INPUT_EVENT_TYPES)
    )

  @property
  def outputs(self) -> PropertyDictWrapper:
    """Dictionary of output artifacts in this component run."""
    return PropertyDictWrapper(
        self._get_artifacts(_OUTPUT_EVENT_TYPES)
    )

  @property
  def exec_properties(self) -> Dict[str, Any]:
    """A dictionary of exec properties of this component."""
    exec_dict = {}
    for key, val in self._execution.properties.items():
      exec_dict[key] = getattr(val, val.WhichOneof('value'))

    for key, val in self._execution.custom_properties.items():
      exec_dict[key] = getattr(val, val.WhichOneof('value'))

    return exec_dict

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

  def __init__(self,
               name: str,
               run_id: str,
               context_id: int,
               store: metadata_store.MetadataStore,
               ir_based_dag_runner: bool = False):
    """Initializes an instance of PipelineRun with an existing run id.

    Args:
      name: The human-readable name of the pipeline.
      run_id: The unique id of the pipeline.
      context_id: The id of the respective context object to this run.
      store: A store for the artifact metadata.
      ir_based_dag_runner: Flag defining whether metadata store was populated by
        an IR based dag runner.
    """
    self.name = name
    self.run_id = run_id
    self._context_id = context_id
    self._store = store
    self._ir_based_dag_runner = ir_based_dag_runner

    if self._ir_based_dag_runner:
      context_type_name = _IR_COMPONENT_NAME
    else:
      context_type_name = _NONIR_COMPONENT_NAME
    self._component_run_type = self._store.get_context_type(context_type_name)

  def __str__(self):
    return 'Pipeline Name: %s\nRun Id: %s' % (self.name, self.run_id)

  def __repr__(self):
    return f'<{self.__str__()}>'

  def _repr_html_(self):
    return pd.DataFrame(
        self.components.keys(), columns=['Components']).to_html(index=False)

  @property
  def components(self) -> Dict[str, ComponentRun]:
    """A dictionary of (Component Name, ComponentRun) key-value pairs."""
    components = {}
    for execution in self._store.get_executions_by_context(self._context_id):
      [component_run_ctx] = [
          ctx for ctx in self._store.get_contexts_by_execution(execution.id)
          if ctx.type_id == self._component_run_type.id
      ]
      if self._ir_based_dag_runner:
        component_name = component_run_ctx.name
      else:
        component_name = execution.properties['component_id'].string_value
      components[component_name] = ComponentRun(self.run_id, execution,
                                                self._store, component_run_ctx)

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

    An Analytics object acts as an accessor to your pipeline runs. You can
    access them as follows:

    analytics = Analytics(mlmd_config)
    analytics.list_pipeline_runs() # [ Pipeline_0, Pipeline_1, ..., Pipeline_N ]
                                   # Ordered by pipeline creation time.

    analytics.get_latest_pipeline_run() # Returns Pipeline_0, where P0 is your
                                        # most recent pipeline by create time.

    analytics.get_pipeline_run(RUN_ID) # Returns pipeline with given run id.

    Args:
      config: Configuration to connect to the database or the metadata store
        server.
      store: A store for the artifact metadata.
    """
    if bool(store) == bool(config):
      raise ValueError('Expected exactly one of store or config')
    self._store = store if store else metadata_store.MetadataStore(config)
    self._ir_based_dag_runner = _IR_COMPONENT_NAME in [
        ctx_type.name for ctx_type in self._store.get_context_types()
    ]

  def _get_pipeline_name(self, ctx):
    """Returns the name of the pipeline associated with ctx."""

    if self._ir_based_dag_runner:
      pipeline_type = self._store.get_context_type('pipeline')
      # The selected execution is arbitrary as all have an association with
      # 'pipeline' context
      execution = self._store.get_executions_by_context(ctx.id)[0]
      [pipeline_ctx] = [
          ctx for ctx in self._store.get_contexts_by_execution(execution.id)
          if ctx.type_id == pipeline_type.id
      ]
      return pipeline_ctx.name
    else:
      return ctx.properties['pipeline_name'].string_value

  def _get_pipeline_runs(self) -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary of runs ids mapped to run and context information."""

    if self._ir_based_dag_runner:
      ctx_name = _IR_RUN_CONTEXT_NAME
    else:
      ctx_name = _NONIR_RUN_CONTEXT_NAME

    ctxs = self._store.get_contexts_by_type(ctx_name)
    runs = {}
    for ctx in ctxs:
      run_id = ctx.name
      pipeline_name = self._get_pipeline_name(ctx)
      runs[run_id] = {
          'pipeline_name': pipeline_name,
          'run_id': run_id,
          'context_id': ctx.id,
          'create_time': ctx.create_time_since_epoch,
          'last_update_time': ctx.last_update_time_since_epoch
      }
    return runs

  def list_pipeline_runs(self) -> List[PipelineRun]:
    """Returns list of pipeline runs in the MLMD store, in order of create time."""
    runs = list(self._get_pipeline_runs().values())
    runs.sort(key=lambda x: x['create_time'], reverse=True)
    return [PipelineRun(run['pipeline_name'], run['run_id'], run['context_id'],
                        self._store, self._ir_based_dag_runner)
            for run in runs]

  def get_pipeline_run(self, run_id: str) -> PipelineRun:
    """Returns a pipeline run object with the given run_id.

    Args:
      run_id: The unique id of the pipeline to query.

    Raises:
      KeyError: If run_id does not exist in the metadata.
    """
    runs = self._get_pipeline_runs()
    if run_id not in runs:
      raise ValueError('Run ID "%s" not found in metadata store.' % run_id)
    return PipelineRun(runs[run_id]['pipeline_name'], run_id,
                       runs[run_id]['context_id'], self._store,
                       self._ir_based_dag_runner)

  def get_latest_pipeline_run(self) -> PipelineRun:
    """Returns the latest pipeline run in the MLMD store.

    Raises:
      ValueError: If there are no pipeline runs in the MLMD store.
    """
    runs = self.list_pipeline_runs()
    if not runs:
      raise ValueError('No pipeline runs found.')
    return runs[0]
