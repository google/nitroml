"""TFX AutoTransform component definition."""

from typing import Any, Dict, Optional, Text, Union

from absl import logging

from components.auto_transform import executor
from tfx import types
from tfx.components.base import base_component, executor_spec
from tfx.orchestration import data_types
from tfx.types import (artifact, artifact_utils, component_spec,
                       standard_artifacts)


class AutoTransformSpec(component_spec.ComponentSpec):
  """AutoTransform Component Spec
    The spec differs from the original Transform Spec in allowing the component
    to receive statistics.
  """

  PARAMETERS = {
      # TODO(nikhilmehta): Where do we want to place the preprocessing_fn for the AutoTransform component. Do we want to expose this fn to users? Or should we just include it as part of the component?
      'module_file':
          component_spec.ExecutionParameter(type=(str, Text), optional=True),
      'custom_config':
          component_spec.ExecutionParameter(
              type=Dict[Text, Any], optional=True)
  }
  INPUTS = {
      'examples':
          component_spec.ChannelParameter(type=standard_artifacts.Examples),
      'schema':
          component_spec.ChannelParameter(type=standard_artifacts.Schema),
      # TODO(nikhilmehta): Add statistics component when we implement autoprocessing.
      # 'statistics':
      #     component_spec.ChannelParameter(
      #         type=standard_artifacts.ExampleStatistics)
  }
  OUTPUTS = {
      'transform_graph':
          component_spec.ChannelParameter(type=standard_artifacts.TransformGraph
                                         ),
      'transformed_examples':
          component_spec.ChannelParameter(type=standard_artifacts.Examples),
  }


class AutoTransform(base_component.BaseComponent):
  """A TFX custom component designed specifically for custom preprocessing.

  Similar to the orginal tfx.components.tranform.Transform, this component will load the prepocessing fn from the input module file.
  However, the signature of the tranforming_fn is as follows:
  def preprocessing_fn(inputs: Dict[Text, Any], schema: schema_pb2.Schema, custom_config: Dict[Text, Any]) -> Dict[Text, Any]

  ## Example usage:
  ```
  tranform = AutoTransform(examples=example_gen.outputs['examples'], schema=infer_schema.outputs['schema'], module_file=module_file)
  ```
  """

  SPEC_CLASS = AutoTransformSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor.AutoTransformExecutor)

  def __init__(
      self,
      examples: types.Channel = None,
      schema: types.Channel = None,
      module_file: Optional[Union[Text, data_types.RuntimeParameter]] = None,
      transform_graph: Optional[types.Channel] = None,
      transformed_examples: Optional[types.Channel] = None,
      input_data: Optional[types.Channel] = None,
      custom_config: Dict[Text, Any] = None,
      instance_name: Optional[Text] = None):

    if input_data:
      logging.warning(
          'The "input_data" argument to the Transform component has '
          'been renamed to "examples" and is deprecated. Please update your '
          'usage as support for this argument will be removed soon.')
      examples = input_data
    if not module_file:
      raise ValueError("'module_file' must be supplied.")

    transform_graph = transform_graph or types.Channel(
        type=standard_artifacts.TransformGraph,
        artifacts=[standard_artifacts.TransformGraph()])
    if not transformed_examples:
      example_artifact = standard_artifacts.Examples()
      example_artifact.split_names = artifact_utils.encode_split_names(
          artifact.DEFAULT_EXAMPLE_SPLITS)
      transformed_examples = types.Channel(
          type=standard_artifacts.Examples, artifacts=[example_artifact])
    spec = AutoTransformSpec(
        examples=examples,
        schema=schema,
        module_file=module_file,
        custom_config=custom_config,
        transform_graph=transform_graph,
        transformed_examples=transformed_examples)
    super(AutoTransform, self).__init__(spec=spec, instance_name=instance_name)
