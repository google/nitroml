"""Testing MLMD object with basic insert functionality."""

from typing import Dict
from nitroml.benchmark import result as br
from nitroml.benchmark import results

from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2


class TestMLMD:
  """TestMLMD abstracts setup and insert operations for a fake mlmd database."""

  _DEFAULT_CONTEXT_PROPERTIES = {
      'pipeline_name': metadata_store_pb2.STRING,
      'run_id': metadata_store_pb2.STRING,
      'component_id': metadata_store_pb2.STRING,
  }

  def __init__(self,
               exec_type_name: str = 'BenchmarkResultPublisher',
               artifact_type: str = br.BenchmarkResult.TYPE_NAME,
               context_type: str = 'run'):
    self.artifact_type = artifact_type
    self.context_type = context_type
    self.config = metadata_store_pb2.ConnectionConfig()
    self.config.fake_database.SetInParent()
    self.store = metadata_store.MetadataStore(self.config)
    self.exec_type_id = self._put_execution_type(exec_type_name)
    self.artifact_type_id = self._put_artifact_type()
    self.context_type_id = self._put_context_type()

  def _put_execution_type(self, exec_type_name: str) -> int:
    exec_type = metadata_store_pb2.ExecutionType()
    exec_type.name = exec_type_name
    exec_type.properties[results.RUN_ID_KEY] = metadata_store_pb2.STRING
    exec_type.properties['component_id'] = metadata_store_pb2.STRING
    return self.store.put_execution_type(exec_type)

  def _put_artifact_type(self) -> int:
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = self.artifact_type
    return self.store.put_artifact_type(artifact_type)

  def _put_context_type(self,
                        properties: Dict[str, metadata_store_pb2.Value] = None
                       ) -> int:
    """Update the context type being inserted by 'put_context'.

    Args:
      properties: Properties to be associated with this context type.

    Returns:
      Id of inserted context
    """
    if not properties:
      # Default propeties correspond with non-IR based MLMD stores.
      properties = self._DEFAULT_CONTEXT_PROPERTIES
    context_type = metadata_store_pb2.ContextType()
    context_type.name = self.context_type
    for name, property_type in properties.items():
      context_type.properties[name] = property_type

    return self.store.put_context_type(context_type)

  def update_context_type(self,
                          context_name: str,
                          properties: Dict[str,
                                           metadata_store_pb2.Value] = None):
    """Update the context type being inserted by 'put_context'.

    Args:
      context_name: Name of the context type to be inserted by 'put_context'.
      properties: Properties to be associated with this context type. Should be
      None for existing context types.
    Raises:
      ValueError: If properties attempt to be associated with an existing
      context type.
    """
    context_types = self.store.get_context_types()
    for ctx_type in context_types:
      if ctx_type.name == context_name:
        if properties:
          raise ValueError(
              'Context type %s already exists, properties cannot be updated' %
              context_name)
        self.context_type = context_name
        self.context_type_id = ctx_type.id
        return
    # Context type does not exist
    self.context_type = context_name
    self.context_type_id = self._put_context_type(properties)

  def put_execution(self, run_id: str, component_id: str = None) -> int:
    """Inserts or Updates an execution into the fake database store.

    Args:
      run_id: The run id of the execution to be inserted or updated.
      component_id: The name of the component which initiated the execution.
    Returns:
      An execution id corresponding with the input.
    """
    execution = metadata_store_pb2.Execution()
    execution.properties[results.RUN_ID_KEY].string_value = run_id
    if component_id:
      execution.properties['component_id'].string_value = component_id
    execution.type_id = self.exec_type_id
    return self.store.put_executions([execution])[0]

  def put_artifact(self, properties: Dict[str, str], name: str = None) -> int:
    """Inserts or updates an artifact in the fake database.

    Args:
      properties: A dictionary of custom properties and values to be added to
        the artifact.
      name: The name of the artifact.

    Returns:
      An artifact id corresponding with the input.
    """
    artifact = metadata_store_pb2.Artifact()
    artifact.uri = 'test/path/'
    artifact.type_id = self.artifact_type_id
    if name:
      artifact.name = name
    for name, val in properties.items():
      artifact.custom_properties[name].string_value = val
    return self.store.put_artifacts([artifact])[0]

  def put_event(
      self,
      artifact_id: int,
      execution_id: int,
      event_type: metadata_store_pb2.Event.Type = metadata_store_pb2.Event
      .OUTPUT
  ) -> None:
    """Inserts event in the fake database.

    The execution_id and artifact_id must already exist.
    Once created, events cannot be modified.

    Args:
      artifact_id: The artifact id of the event.
      execution_id: The execution id of the event.
      event_type: The type of the event.
    """
    event = metadata_store_pb2.Event()
    event.type = event_type
    event.artifact_id = artifact_id
    event.execution_id = execution_id
    self.store.put_events([event])

  def put_context(self,
                  name: str = None,
                  context_id: int = None,
                  properties: Dict[str, str] = None,
                  context: metadata_store_pb2.Context = None) -> int:
    """Inserts or updates contexts in the fake database.

    If an context_id is specified for an context, it is an update.
    If an context_id is unspecified, it will insert a new context.
    For new contexts, type must be specified.
    For old contexts, type must be unchanged or unspecified.
    The name of a context cannot be empty, and it should be unique among
    contexts of the same ContextType.

    Args:
      name: The name of the context to be inserted or updated.
      context_id: The unique id of the context to be inserted or updated.
      properties: Dict of properties to add to the context.
      context: A context proto to be inserted. If defined, name, context_id,
      and properties are ignored.

    Returns:
      The unique id of the context to be inserted or updated.
    """
    if context:
      context.type = self.context_type
      context.type_id = self.context_type_id
    else:
      context = metadata_store_pb2.Context()
      context.name = name
      if context_id:  # Updating a context with given id
        context.id = context_id
      else:  # Inserting a new context
        context.type = self.context_type
        context.type_id = self.context_type_id

      if properties:
        for name, val in properties.items():
          context.properties[name].string_value = val

    return self.store.put_contexts([context])[0]

  def put_association(self, context_id: int, execution_id: int):
    """Inserts association relationship in the fake database.

    The context_id and execution_id must already exist.
    If the relationship exists, this call does nothing. Once added, the
    relationships cannot be modified.

    Args:
      context_id: The unique id of the context to be associated.
      execution_id: The unique id of the execution to be associated.
    """
    association = metadata_store_pb2.Association()
    association.context_id = context_id
    association.execution_id = execution_id
    self.store.put_attributions_and_associations([], [association])

  def put_attribution(self, context_id: int, artifact_id: int):
    """Inserts attribution relationship in the fake database.

    The context_id and artifact_id must already exist.
    If the relationship exists, this call does nothing. Once added, the
    relationships cannot be modified.

    Args:
      context_id: The unique id of the context to be attributed.
      artifact_id: The unique id of the artifact to be attributed.
    """
    attribution = metadata_store_pb2.Attribution()
    attribution.context_id = context_id
    attribution.artifact_id = artifact_id
    self.store.put_attributions_and_associations([attribution], [])
