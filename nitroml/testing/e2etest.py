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
"""Base class for writing end-to-end tests for NitroML benchmarks.

This is NOT meant for testing TFX components; it is meant to provide
users a way to run their benchmark pipelines to quickly validate both
their user code and their pipelines.

For an example, see
nitroml/examples/titanic_benchmark_test.py.
"""

import os
import sys
import tempfile
from typing import List

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import nitroml
import tensorflow as tf
from tfx.dsl.compiler import constants
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_dag_runner

from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS

main = absltest.main


class TestCase(parameterized.TestCase, absltest.TestCase):
  """Base class for end-to-end NitroML benchmark tests."""

  @property
  def pipeline_name(self) -> str:
    """Returns the test pipeline's name."""

    return "test_pipeline"

  def setUp(self):
    """Sets up the end-to-end test.

    Creates the pipeline_root directories, and the metadata instance as a SQLite
    instance.
    """

    super(TestCase, self).setUp()

    FLAGS(sys.argv)  # Required for tests that use flags in open-source.
    tempdir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self.pipeline_root = os.path.join(tempdir, self.pipeline_name)
    tf.io.gfile.makedirs(self.pipeline_root)
    logging.info("pipeline_root located at %s", self.pipeline_root)
    self.metadata_path = os.path.join(self.pipeline_root, "mlmd.sqlite")
    logging.info("MLMD SQLite instance located at %s", self.metadata_path)

  @property
  def metadata_config(self) -> metadata_store_pb2.ConnectionConfig:
    return metadata_store_pb2.ConnectionConfig(
        sqlite=metadata_store_pb2.SqliteMetadataSourceConfig(
            filename_uri=self.metadata_path))

  def run_pipeline(self,
                   components: List[base_component.BaseComponent]) -> None:
    """Creates and runs a pipeline with the given components."""

    runner = beam_dag_runner.BeamDagRunner()
    runner.run(
        pipeline.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            metadata_connection_config=self.metadata_config,
            beam_pipeline_args=[
            ],
            components=components))

  def assertComponentSucceeded(self, component_name: str) -> None:
    """Checks that the component succeeded.

    Args:
      component_name: Name of the component to check, e.g. 'SchemaGen'.
    """

    self.assertTrue(tf.io.gfile.exists(self.metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
        self.metadata_path)

    with metadata.Metadata(metadata_config) as m:
      node_context_type = m.store.get_context_type(
          constants.NODE_CONTEXT_TYPE_NAME)

      component_states = {}
      for execution in m.store.get_executions():
        node_name = None
        for context in m.store.get_contexts_by_execution(execution.id):
          if context.type_id == node_context_type.id:
            node_name = context.name
            break

        assert node_name is not None

        # NOTE: The node name is '<pipeline_name>.<component_name>'.
        exec_component_name = node_name.replace(f"{self.pipeline_name}.", "")
        component_states[exec_component_name] = execution.last_known_state

    if component_name in component_states:
      self.assertEqual(3, component_states[component_name])  # COMPLETE = 3
      return
    all_components = sorted(component_states.keys())
    raise ValueError(
        f'Failed to find component "{component_name}". Found {all_components}')

  def assertComponentsSucceeded(self, component_names: List[str]) -> None:
    """Checks that the components succeeded.

    Args:
      component_names: Names of the components to check, e.g. 'SchemaGen'.
    """

    for name in component_names:
      self.assertComponentSucceeded(name)

  def assertComponentExecutionCount(self, count: int) -> None:
    """Checks the number of component executions recorded in MLMD.

    Args:
      count: Number of components that should have succeeded and produced
        artifacts recorded in MLMD.
    """

    self.assertTrue(tf.io.gfile.exists(self.metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
        self.metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(count, execution_count)

  def artifact_dir(self, artifact_root: str, artifact_subdir: str = "") -> str:
    """Returns the full path to the artifact subdir under the pipeline root.

    For example to get the transformed examples for the train split:

      `self.artifact_dir('Transform.AutoData/transformed_examples', 'train/*')`

    would return the path

      '<pipeline_root>/Transform.AutoData/transformed_examples/4/train/*'.

    Assumes there is a single execution per component.

    Args:
      artifact_root: Root subdirectory to specify the component's artifacts.
      artifact_subdir: Optional subdirectory to append after the execution
        ID.

    Returns:
      The full path to the artifact subdir under the pipeline root.
    """

    root = os.path.join(self.pipeline_root, artifact_root)
    artifact_subdirs = tf.io.gfile.listdir(root)
    if len(artifact_subdirs) != 1:
      raise ValueError(
          f"Expected a single artifact dir, got: {artifact_subdirs}")
    base_uri = base_driver._generate_output_uri(  # pylint: disable=protected-access
        self.pipeline_root, artifact_root, artifact_subdirs[0])
    return os.path.join(base_uri, artifact_subdir)


class BenchmarkTestCase(TestCase):
  """A test case specifically for testing NitroML benchmarks."""

  def run_benchmarks(self, benchmarks: List[nitroml.Benchmark],
                     **kwargs) -> None:
    """Runs the given benchmarks with nitroml using a BeamDagRunner.

    Args:
      benchmarks: List of `nitroml.Benchmark` to run.
      **kwargs: Keyword args to pass to `nitroml#run`.
    """

    nitroml.run(
        benchmarks,
        pipeline_name=kwargs.pop("pipeline_name", self.pipeline_name),
        pipeline_root=kwargs.pop("pipeline_root", self.pipeline_root),
        metadata_connection_config=kwargs.pop("metadata_connection_config",
                                              self.metadata_config),
        tfx_runner=kwargs.pop("tfx_runner", beam_dag_runner.BeamDagRunner()),
        beam_pipeline_args=kwargs.pop(
            "beam_pipeline_args",
            [
            ]),
        **kwargs)
