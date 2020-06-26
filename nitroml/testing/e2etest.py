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

from typing import Text

from absl.testing import absltest
import tensorflow as tf

from tfx.orchestration import metadata


class TestCase(absltest.TestCase):
  """Base class for end-to-end NitroML benchmark tests."""

  def assertComponentSucceeded(self, component: Text,
                               metadata_path: Text) -> None:
    """Checks that the component succeeded.

    Args:
      component: Name of the component to check, e.g. 'SchemaGen'.
      metadata_path: Path to the MLMD store, e.g.
        '/tmp/nitroml_benchmark/mlmd.sqlite'.
    """

    self.assertTrue(tf.io.gfile.exists(metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(metadata_path)

    component_states = {}
    with metadata.Metadata(metadata_config) as m:
      for execution in m.store.get_executions():
        component_name = execution.properties['component_id'].string_value
        state = execution.properties['state'].string_value
        component_states[component_name] = state
    if component in component_states:
      self.assertEqual('complete', component_states[component])
      return
    all_components = sorted(component_states.keys())
    raise ValueError(
        f'Failed to find component "{component}". Found {all_components}')

  def assertComponentExecutionCount(self, count: int,
                                    metadata_path: Text) -> None:
    """Checks the number of component executions recorded in MLMD.

    Args:
      count: Number of components that should have succeeded and produced
        artifacts recorded in MLMD.
      metadata_path: Path to the MLMD store, e.g.
        '/tmp/nitroml_benchmark/mlmd.sqlite'.
    """

    self.assertTrue(tf.io.gfile.exists(metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(count, execution_count)
