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

from absl.testing import absltest

from nitroml.analytics import materialized_artifact
from tfx import types
from tfx.types import standard_artifacts
from ml_metadata.proto import metadata_store_pb2


class MaterializedArtifactTest(absltest.TestCase):

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
    self.assertEqual(
        '<Examples Artifact>',
        repr(
            materialized_artifact.MaterializedArtifact(
                types.Artifact(mlmd_artifact_type=artifact_type))))


if __name__ == '__main__':
  absltest.main()
