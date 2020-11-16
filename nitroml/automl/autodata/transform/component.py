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
"""A Transform TFX component that support a more flexible preprocessing_fns."""

import json
from typing import Any, Dict, Optional, Text, Union

from nitroml.automl.autodata.transform import executor
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


def noop_preprocessing_fn(inputs):
  """No-op preprocessing fn for Transform.

  This is provided as a convenience for pipelines that don't do any data
  preprocessing but require a Transform component anyway (e.g. because the
  trainer expects a transform graph as input).

  Args:
    inputs: Dict of input tensors.

  Returns:
    The unmodified `inputs` dict.
  """
  return inputs


class Transform(base_component.BaseComponent):
  """Custom TFX Transform component which allows richer 'preprocessing_fns'.

  Functions the same exact way as the OSS TFX Transform component but extends
  the 'preprocessing_fn' to optionally declare 2 new arguments: 'schema' and
  'custom_config'.

  Examples:
    def preprocessing_fn(inputs, schema, custom_config):
      problem_statement_path = custom_config['problem_statement_path']
      ...

    def preprocessing_fn(inputs):
      ...

  TODO(github.com/tensorflow/tfx/issues/687): Remove this once TFX-core accepts
  a more flexible preprocessing_fn.
  """

  SPEC_CLASS = standard_component_specs.TransformSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      schema: types.Channel = None,
      module_file: Optional[Union[Text, data_types.RuntimeParameter]] = None,
      preprocessing_fn: Optional[Union[Text,
                                       data_types.RuntimeParameter]] = None,
      custom_config: Optional[Dict[Text, Any]] = None,
      transform_graph: Optional[types.Channel] = None,
      transformed_examples: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    # pyformat: disable
    # pylint: disable=g-doc-args
    """Construct a Transform component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples` (required).
        This should contain the two splits 'train' and 'eval'.
      schema: A Channel of type `standard_artifacts.Schema`. This should
        contain a single schema artifact.
      module_file: The file path to a python module file, from which the
        'preprocessing_fn' function will be loaded. The function must have the
        following signature.

        def preprocessing_fn(inputs: Dict[Text, Any],
                             schema: schema_pb2.Schema,
                             custom_config: Dict[Text, Any]) -> Dict[Text, Any]:
          ...

        where the values of input and returned Dict are either tf.Tensor or
        tf.SparseTensor. The 'schema' and 'custom_config' arguments are not
        necessary and can be omitted. Exactly one of 'module_file' or
        'preprocessing_fn' must be supplied.
      preprocessing_fn: The path to python function that implements a
        'preprocessing_fn'. See 'module_file' for expected signature of the
        function. Exactly one of 'module_file' or 'preprocessing_fn' must be
        supplied.
      custom_config: A dict which contains additional transform parameters that
        will be passed into the preprocessing_fn.
      transform_graph: Optional output 'TransformPath' channel for output of
        'tf.Transform', which includes an exported Tensorflow graph suitable for
        both training and serving;
      transformed_examples: Optional output 'ExamplesPath' channel for
        materialized transformed examples, which includes both 'train' and
        'eval' splits.
      instance_name: Optional unique instance name. Necessary iff multiple
        transform components are declared in the same pipeline.

    Raises:
      ValueError: When both or neither of 'module_file' and 'preprocessing_fn'
        is supplied.
    """
    # pyformat: enable
    # pylint: enable=g-doc-args
    if bool(module_file) == bool(preprocessing_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'preprocessing_fn' must be supplied."
      )
    transform_graph = transform_graph or types.Channel(
        type=standard_artifacts.TransformGraph,
        artifacts=[standard_artifacts.TransformGraph()])
    if not transformed_examples:
      example_artifact = standard_artifacts.Examples()
      example_artifact.split_names = artifact_utils.encode_split_names(
          artifact.DEFAULT_EXAMPLE_SPLITS)
      transformed_examples = types.Channel(
          type=standard_artifacts.Examples, artifacts=[example_artifact])
    spec = standard_component_specs.TransformSpec(
        examples=examples,
        schema=schema,
        module_file=module_file,
        preprocessing_fn=preprocessing_fn,
        custom_config=json.dumps(custom_config),
        transform_graph=transform_graph,
        transformed_examples=transformed_examples)
    super(Transform, self).__init__(spec=spec, instance_name=instance_name)
