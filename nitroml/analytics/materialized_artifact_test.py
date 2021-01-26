# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for nitroml.analytics.materialized_artifact."""

from typing import Any, Dict, Mapping
from absl.testing import absltest
from nitroml.analytics import materialized_artifact
from tfx import types
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2


class MaterializedArtifactTest(absltest.TestCase):

  def _unpack_properties(
      self, artifact_properties: Mapping[str, Any]) -> Dict[str, str]:
    return {k: v.string_value for k, v in artifact_properties.items()}

  def testArtifactRegistry(self):
    registry = materialized_artifact.ArtifactRegistry()

    class MyArtifact(materialized_artifact.MaterializedArtifact):

      # Arbitrary artifact type class.
      ARTIFACT_TYPE = standard_artifacts.Examples

    registry.register(MyArtifact)
    self.assertIs(
        MyArtifact,
        registry.get_artifact_class(standard_artifacts.Examples.TYPE_NAME))

  def testRepr(self):
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = 'Examples'
    tfx_artifact = types.Artifact(mlmd_artifact_type=artifact_type)
    tfx_artifact.uri = 'test/uri'
    want = '<Type: Examples, URI: test/uri>'
    got = repr(materialized_artifact.GenericMaterializedArtifact(tfx_artifact))
    self.assertEqual(want, got)

  def testProperties(self):
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = 'Examples'
    artifact_type.id = 1
    tfx_artifact = types.Artifact(mlmd_artifact_type=artifact_type)

    artifact = metadata_store_pb2.Artifact(
        uri='test/uri', id=2, name='test_artifact', type_id=1)
    tfx_artifact.set_mlmd_artifact(artifact)
    tfx_artifact.producer_component = 'test_producer'
    ma = materialized_artifact.GenericMaterializedArtifact(tfx_artifact)

    self.assertEqual('test/uri', ma.uri)
    self.assertEqual(2, ma.id)
    self.assertEqual('test_artifact', ma.name)
    self.assertEqual('test_producer', ma.producer_component)
    self.assertEqual('Examples', ma.type_name)
    self.assertEqual(self._unpack_properties(artifact.custom_properties),
                     ma.properties)
    with self.assertRaises(NotImplementedError):
      ma.show()


if __name__ == '__main__':
  absltest.main()
