# Copyright 2021 Google LLC
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
r"""AutoData Python callable components."""

import os

from absl import logging
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types import standard_artifacts
from tfx.utils import io_utils

from tensorflow_metadata.proto.v0 import schema_pb2


# pytype: disable=wrong-arg-types
@component
def annotate_schema(
    ignore_features: Parameter[str],
    original_schema: InputArtifact[standard_artifacts.Schema],
    schema: OutputArtifact[standard_artifacts.Schema],
) -> None:  # pytype: disable=invalid-annotation,wrong-arg-types
  r"""Updates a schema with additional metadata.

  Args:
    ignore_features: Newline ('\n') separated list of features to mark as
      disabled in the output schema.
    original_schema: The Schema artifact to modify.
    schema: The output Schema with updates.
  """

  schema_file = io_utils.get_only_uri_in_dir(original_schema.uri)
  dataset_schema = schema_pb2.Schema()
  io_utils.parse_pbtxt_file(schema_file, dataset_schema)

  ignore_features = ignore_features.split("\n")
  for feature in dataset_schema.feature:
    if feature.name in ignore_features:
      logging.info("Marking '%s' as DISABLED.", feature.name)
      feature.lifecycle_stage = schema_pb2.LifecycleStage.DISABLED

  io_utils.write_pbtxt_file(
      os.path.join(schema.uri, "schema.txt"), dataset_schema)


# pytype: enable=wrong-arg-types
