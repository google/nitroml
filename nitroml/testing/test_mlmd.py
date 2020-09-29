"""Testing MLMD object with basic insert functionality."""

from typing import Dict
from nitroml import results

from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2


class TestMLMD:
  """TestMLMD abstracts setup and insert operations for a fake mlmd database."""

  def __init__(self,
               exec_type_name: str = 'BenchmarkResultPublisher',
               artifact_type: str = results._BENCHMARK_RESULT):
    self.artifact_type = artifact_type
    config = metadata_store_pb2.ConnectionConfig()
    config.fake_database.SetInParent()
    self.store = metadata_store.MetadataStore(config)
    self.exec_type_id = self._put_execution_type(exec_type_name)
    self.artifact_type_id = self._put_artifact_type()

  def _put_execution_type(self, exec_type_name: str) -> int:
    exec_type = metadata_store_pb2.ExecutionType()
    exec_type.name = exec_type_name
    exec_type.properties[results.RUN_ID_KEY] = metadata_store_pb2.STRING
    return self.store.put_execution_type(exec_type)

  def _put_artifact_type(self) -> int:
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = self.artifact_type
    return self.store.put_artifact_type(artifact_type)

  def put_execution(self, run_id: str) -> int:
    """Inserts or Updates an execution into the fake database store.

    Args:
      run_id: The run id of the execution to be inserted or updated.

    Returns:
      An execution id corresponding with the input.
    """
    execution = metadata_store_pb2.Execution()
    execution.properties[results.RUN_ID_KEY].string_value = run_id
    execution.type_id = self.exec_type_id
    return self.store.put_executions([execution])[0]

  def put_artifact(self, properties: Dict[str, str]) -> int:
    """Inserts or updates an artifact in the fake database.

    Args:
      properties: A dictionary of custom properties and values to be added to
        the artifact.

    Returns:
      An artifact id corresponding with the input.
    """
    artifact = metadata_store_pb2.Artifact()
    artifact.uri = 'test/path/'
    artifact.type_id = self.artifact_type_id
    for name, val in properties.items():
      artifact.custom_properties[name].string_value = val
    return self.store.put_artifacts([artifact])[0]

  def put_event(self, artifact_id: int, execution_id: int) -> None:
    """Inserts event in the fake database.

    The execution_id and artifact_id must already exist.
    Once created, events cannot be modified.

    Args:
      artifact_id: The artifact id of the event.
      execution_id: The execution id of the event.
    """
    event = metadata_store_pb2.Event()
    event.type = metadata_store_pb2.Event.OUTPUT
    event.artifact_id = artifact_id
    event.execution_id = execution_id
    self.store.put_events([event])
