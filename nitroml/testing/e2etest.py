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

import json
import os
import tempfile
from typing import List, Text

from absl import flags
from absl.testing import absltest
import nitroml
import tensorflow as tf
from tfx.components.base import base_component
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_dag_runner

from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS

main = absltest.main


class TestCase(absltest.TestCase):
  """Base class for end-to-end NitroML benchmark tests."""

  def setUp(self, pipeline_name: Text):
    """Sets up the end-to-end test.

    Creates the pipeline_root directories, and the metadata instance as a SQLite
    instance.

    Args:
      pipeline_name: String pipeline name.
    """
    super(TestCase, self).setUp()

    tempdir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self.pipeline_name = "autodata_test"
    self.pipeline_root = os.path.join(tempdir, self.pipeline_name)
    tf.io.gfile.makedirs(self.pipeline_root)
    self.metadata_path = os.path.join(self.pipeline_root, "metadata.db")

    # BEGIN GOOGLE-INTERNAL
    # Required for the MyOrchestratorRunner when run in google3.
    FLAGS.command = "launch"
    FLAGS.config = json.dumps({
        "pipeline_name": pipeline_name,
        "pipeline_root": self.pipeline_root,
        "pipeline_args": {},
        "enable_cache": False,
        "metadata_connection_config": {
            "sqlite": {
                "filename_uri": self.metadata_path,
            }
        },
        "local_config": {},
        "beam_pipeline_args": [
            "--runner=google3.pipeline.flume.py.runner.FlumeRunner"
        ],
        "flume_args": ["--flume_exec_mode=UNOPT"],
        "launch_config": {},
    })
    # BEGIN GOOGLE-INTERNAL

  @property
  def metadata_config(self):
    return metadata_store_pb2.ConnectionConfig(
        sqlite=metadata_store_pb2.SqliteMetadataSourceConfig(
            filename_uri=self.metadata_path))

  def run_pipeline(self, components: List[base_component.BaseComponent]):
    """Creates and runs a pipeline with the given components."""

    runner = beam_dag_runner.BeamDagRunner()
    runner.run(
        pipeline.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            metadata_connection_config=self.metadata_config,
            components=components))

  def run_benchmarks(self, benchmarks: List[nitroml.Benchmark], **kwargs):
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
        **kwargs)

  def assertComponentSucceeded(self, component_name: Text) -> None:
    """Checks that the component succeeded.

    Args:
      component_name: Name of the component to check, e.g. 'SchemaGen'.
    """

    self.assertTrue(tf.io.gfile.exists(self.metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
        self.metadata_path)

    component_states = {}
    with metadata.Metadata(metadata_config) as m:
      for execution in m.store.get_executions():
        component_id = execution.properties["component_id"].string_value
        state = execution.properties["state"].string_value
        component_states[component_id] = state
    if component_name in component_states:
      self.assertEqual("complete", component_states[component_name])
      return
    all_components = sorted(component_states.keys())
    raise ValueError(
        f'Failed to find component "{component_name}". Found {all_components}')

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
