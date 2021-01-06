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
"""Tests for nitroml.pipeline_filtering."""

from typing import Any, Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from nitroml import pipeline_filtering

from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2 as p_pb2
from tfx.utils import test_case_utils
from google3.google.protobuf import any_pb2
from google3.learning.tfx.tflex.proto.deployment_config.pluggable_orchestrator import deployment_config_pb2
from ml_metadata.proto import metadata_store_pb2 as mlmd_pb2


def to_context_spec(type_name: str, name: str) -> p_pb2.ContextSpec:
  return p_pb2.ContextSpec(
      type=mlmd_pb2.ContextType(name=type_name),
      name=p_pb2.Value(field_value=mlmd_pb2.Value(string_value=name)))


def to_output_spec(artifact_name: str) -> p_pb2.OutputSpec:
  return p_pb2.OutputSpec(
      artifact_spec=p_pb2.OutputSpec.ArtifactSpec(
          type=mlmd_pb2.ArtifactType(name=artifact_name)))


def to_input_channel(
    producer_output_key: str, producer_node_id: str, artifact_type: str,
    context_names: Mapping[str, str]) -> p_pb2.InputSpec.Channel:
  # pylint: disable=g-complex-comprehension
  context_queries = [
      p_pb2.InputSpec.Channel.ContextQuery(
          type=mlmd_pb2.ContextType(name=context_type),
          name=p_pb2.Value(
              field_value=mlmd_pb2.Value(string_value=context_name)))
      for context_type, context_name in context_names.items()
  ]
  # pylint: enable=g-complex-comprehension
  return p_pb2.InputSpec.Channel(
      output_key=producer_output_key,
      producer_node_query=p_pb2.InputSpec.Channel.ProducerNodeQuery(
          id=producer_node_id),
      context_queries=context_queries,
      artifact_query=p_pb2.InputSpec.Channel.ArtifactQuery(
          type=mlmd_pb2.ArtifactType(name=artifact_type)))


def to_any_proto(input_proto):
  result = any_pb2.Any()
  result.Pack(input_proto)
  return result


def make_dummy_executable_specs(node_ids: Sequence[str]) -> Mapping[str, Any]:
  result = {}
  for node_id in node_ids:
    result[node_id] = to_any_proto(
        deployment_config_pb2.ExecutableSpec(
            python_class_executable_spec=(
                executable_spec_pb2.PythonClassExecutableSpec(
                    class_path='google3.path.to.Executable',
                    extra_flags=[node_id, 'extra', 'flags']))))
  return result


def make_dummy_custom_driver_specs(
    node_ids: Sequence[str]) -> Mapping[str, Any]:
  result = {}
  for node_id in node_ids:
    result[node_id] = to_any_proto(
        deployment_config_pb2.ExecutableSpec(
            python_class_executable_spec=(
                executable_spec_pb2.PythonClassExecutableSpec(
                    class_path='google3.path.to.CustomDriver',
                    extra_flags=[node_id, 'extra', 'flags']))))
  return result


def make_dummy_node_level_platform_configs(
    node_ids: Sequence[str]) -> Mapping[str, Any]:
  result = {}
  for node_id in node_ids:
    result[node_id] = to_any_proto(
        deployment_config_pb2.BorgPlatformConfig(
            logs_read_access_roles=f'{node_id}.logreader'))
  return result


class PipelineFilteringTest(parameterized.TestCase, test_case_utils.TfxTest):

  def testSubpipeline_error(self):
    """If Pipeline contains sub-pipeline, raise NotImplementedError."""
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ])))
    sub_pipeline_node = p_pb2.Pipeline.PipelineOrNode(
        sub_pipeline=p_pb2.Pipeline(
            pipeline_info=p_pb2.PipelineInfo(id='my_subpipeline'),
            execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
            nodes=[node_a]))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[sub_pipeline_node])

    with self.assertRaises(ValueError):
      _ = pipeline_filtering.filter_pipeline(
          input_pipeline,
          from_nodes=lambda _: True,
          to_nodes=lambda _: True,
          old_pipeline_run_id='pipeline_run_000')

  def testNoFilter(self):
    """Basic case where there are no filters applied.

    input_pipeline: node_a -> node_b -> node_c
    from_node: all nodes
    to_node: all nodes
    expected output_pipeline: node_a -> node_b -> node_c
    """

    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline = pipeline_filtering.filter_pipeline(
        input_pipeline,
        from_nodes=lambda _: True,
        to_nodes=lambda _: True,
        old_pipeline_run_id='pipeline_run_000')

    self.assertProtoEquals(input_pipeline, filtered_pipeline)

  def testFilterOutNothing(self):
    """Basic case where no nodes are filtered out.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_a
    to_node: node_c
    expected output_pipeline: node_a -> node_b -> node_c
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline = pipeline_filtering.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'c'),
        old_pipeline_run_id='pipeline_run_000')

    self.assertProtoEquals(input_pipeline, filtered_pipeline)

  def testFilterOutSinkNode(self):
    """Filter out a node that has upstream nodes but no downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    to_node: node_b
    expected_output_pipeline: node_a -> node_b
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline = pipeline_filtering.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'b'),
        old_pipeline_run_id='pipeline_run_000')

    node_b_no_downstream = p_pb2.Pipeline.PipelineOrNode()
    node_b_no_downstream.CopyFrom(node_b)
    del node_b_no_downstream.pipeline_node.downstream_nodes[:]
    expected_output_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b_no_downstream])
    self.assertProtoEquals(expected_output_pipeline, filtered_pipeline)

  def testFilterOutSourceNode(self):
    """Filter out a node that has no upstream nodes but has downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_b
    to_node: node_c
    old_pipeline_run_id: pipeline_run_000
    expected_output_pipeline: node_b -> node_c
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline = pipeline_filtering.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'),
        old_pipeline_run_id='pipeline_run_000')

    node_b_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_b_fixed.CopyFrom(node_b)
    del node_b_fixed.pipeline_node.upstream_nodes[:]
    node_b_fixed.pipeline_node.inputs.inputs['in'].channels[
        0].context_queries.append(
            p_pb2.InputSpec.Channel.ContextQuery(
                type=mlmd_pb2.ContextType(name='pipeline_run'),
                name=p_pb2.Value(
                    field_value=mlmd_pb2.Value(
                        string_value='pipeline_run_000'))))
    expected_output_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_b_fixed, node_c])
    self.assertProtoEquals(expected_output_pipeline, filtered_pipeline)

  def testFilterOutSourceNode_triangle(self):
    """Filter out a source node in a triangle.

    input_pipeline:
        node_a -> node_b -> node_c
             |--------------^
    from_node: node_b
    to_node: node_c
    old_pipeline_run_id: pipeline_run_000
    expected_output_pipeline: node_b -> node_c
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={
                'out_b': to_output_spec('AB'),
                'out_c': to_output_spec('AC')
            }),
            downstream_nodes=['b', 'c']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in_a':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out_c',
                                    artifact_type='AC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1),
                    'in_b':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['a', 'b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline = pipeline_filtering.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'),
        old_pipeline_run_id='pipeline_run_000')

    node_b_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_b_fixed.CopyFrom(node_b)
    del node_b_fixed.pipeline_node.upstream_nodes[:]
    node_b_fixed.pipeline_node.inputs.inputs['in'].channels[
        0].context_queries.append(
            p_pb2.InputSpec.Channel.ContextQuery(
                type=mlmd_pb2.ContextType(name='pipeline_run'),
                name=p_pb2.Value(
                    field_value=mlmd_pb2.Value(
                        string_value='pipeline_run_000'))))
    node_c_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_c_fixed.CopyFrom(node_c)
    node_c_fixed.pipeline_node.upstream_nodes[:] = 'b'
    node_c_fixed.pipeline_node.inputs.inputs['in_a'].channels[
        0].context_queries.append(
            p_pb2.InputSpec.Channel.ContextQuery(
                type=mlmd_pb2.ContextType(name='pipeline_run'),
                name=p_pb2.Value(
                    field_value=mlmd_pb2.Value(
                        string_value='pipeline_run_000'))))
    expected_output_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_b_fixed, node_c_fixed])
    self.assertProtoEquals(expected_output_pipeline, filtered_pipeline)

  def testSkipNodes(self):
    """Skip a node in the middle.

    input_pipeline: node_a -> node_b -> node_c
    from_node: all nodes
    to_node: all nodes
    skip_node: node_b
    old_pipeline_run_id: pipeline_run_000
    expected_output_pipeline: node_a (unconnected) node_c
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline = pipeline_filtering.filter_pipeline(
        input_pipeline,
        skip_nodes=lambda node_id: (node_id == 'b'),
        old_pipeline_run_id='pipeline_run_000')

    node_a_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_a_fixed.CopyFrom(node_a)
    del node_a_fixed.pipeline_node.downstream_nodes[:]
    node_c_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_c_fixed.CopyFrom(node_c)
    del node_c_fixed.pipeline_node.upstream_nodes[:]
    node_c_fixed.pipeline_node.inputs.inputs['in'].channels[
        0].context_queries.append(
            p_pb2.InputSpec.Channel.ContextQuery(
                type=mlmd_pb2.ContextType(name='pipeline_run'),
                name=p_pb2.Value(
                    field_value=mlmd_pb2.Value(
                        string_value='pipeline_run_000'))))
    expected_output_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a_fixed, node_c_fixed])
    self.assertProtoEquals(expected_output_pipeline, filtered_pipeline)

  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'input_deployment_cfg': None,
          'expected_deployment_cfg': any_pb2.Any()
      }, {
          'testcase_name':
              'all',
          'input_deployment_cfg':
              to_any_proto(
                  p_pb2.IntermediateDeploymentConfig(
                      executor_specs=make_dummy_executable_specs('abc'),
                      custom_driver_specs=make_dummy_custom_driver_specs('abc'),
                      node_level_platform_configs=(
                          make_dummy_node_level_platform_configs('abc')),
                      metadata_connection_config=to_any_proto(
                          mlmd_pb2.ConnectionConfig(
                              fake_database=mlmd_pb2.FakeDatabaseConfig())))),
          'expected_deployment_cfg':
              to_any_proto(
                  p_pb2.IntermediateDeploymentConfig(
                      executor_specs=make_dummy_executable_specs('ab'),
                      custom_driver_specs=make_dummy_custom_driver_specs('ab'),
                      node_level_platform_configs=(
                          make_dummy_node_level_platform_configs('ab')),
                      metadata_connection_config=to_any_proto(
                          mlmd_pb2.ConnectionConfig(
                              fake_database=mlmd_pb2.FakeDatabaseConfig()))))
      }, {
          'testcase_name':
              'missing_fields',
          'input_deployment_cfg':
              to_any_proto(
                  p_pb2.IntermediateDeploymentConfig(
                      executor_specs=make_dummy_executable_specs('abc'),
                      metadata_connection_config=to_any_proto(
                          mlmd_pb2.ConnectionConfig(
                              fake_database=mlmd_pb2.FakeDatabaseConfig())))),
          'expected_deployment_cfg':
              to_any_proto(
                  p_pb2.IntermediateDeploymentConfig(
                      executor_specs=make_dummy_executable_specs('ab'),
                      metadata_connection_config=to_any_proto(
                          mlmd_pb2.ConnectionConfig(
                              fake_database=mlmd_pb2.FakeDatabaseConfig()))))
      }, {
          'testcase_name':
              'different_fields',
          'input_deployment_cfg':
              to_any_proto(
                  p_pb2.IntermediateDeploymentConfig(
                      executor_specs=make_dummy_executable_specs('c'),
                      custom_driver_specs=make_dummy_custom_driver_specs('bc'),
                      node_level_platform_configs=(
                          make_dummy_node_level_platform_configs('ab')),
                      metadata_connection_config=to_any_proto(
                          mlmd_pb2.ConnectionConfig(
                              fake_database=mlmd_pb2.FakeDatabaseConfig())))),
          'expected_deployment_cfg':
              to_any_proto(
                  p_pb2.IntermediateDeploymentConfig(
                      custom_driver_specs=make_dummy_custom_driver_specs('b'),
                      node_level_platform_configs=(
                          make_dummy_node_level_platform_configs('ab')),
                      metadata_connection_config=to_any_proto(
                          mlmd_pb2.ConnectionConfig(
                              fake_database=mlmd_pb2.FakeDatabaseConfig()))))
      })
  def testDeploymentConfig(self, input_deployment_cfg, expected_deployment_cfg):
    """Test that per-node deployment configs are filtered correctly."""
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=mlmd_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                to_context_spec('pipeline', 'my_pipeline'),
                to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c],
        deployment_config=input_deployment_cfg)

    filtered_pipeline = pipeline_filtering.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'b'),
        old_pipeline_run_id='pipeline_run_000')

    self.assertProtoEquals(expected_deployment_cfg,
                           filtered_pipeline.deployment_config)


if __name__ == '__main__':
  absltest.main()
