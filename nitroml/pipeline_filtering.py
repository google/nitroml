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
"""Pipeline filtering.

Allow users to create a Pipeline that consists of a subset of the nodes from
a given Pipeline.
"""

import collections
import enum
from typing import Any, Callable, Collection, Mapping, MutableMapping, Optional, Set, Union

from nitroml.analytics import mlmd_analytics
from tfx.dsl.compiler import constants as dsl_constants
from tfx.proto.orchestration import pipeline_pb2 as p_pb2

from google3.google.protobuf import any_pb2
from ml_metadata.proto import metadata_store_pb2 as mlmd_pb2


def filter_pipeline(
    input_pipeline: p_pb2.Pipeline,
    pipeline_run_id_fn: Callable[[p_pb2.InputSpec.Channel], str],
    from_nodes: Optional[Callable[[str], bool]] = None,
    to_nodes: Optional[Callable[[str], bool]] = None,
    skip_nodes: Optional[Callable[[str], bool]] = None,
) -> p_pb2.Pipeline:
  """Filters the Pipeline IR proto, thus enabling partial runs.

  The set of nodes included in the filtered pipeline is the set of nodes between
  from_nodes and to_nodes, minus the set of skip_nodes. Note that the
  input_pipeline will not have any subpipeline nodes, since the compiler is
  supposed to flatten them. Also, if the input_pipeline contains per-node
  DeploymentConfigs, they will be filtered as well.

  Args:
    input_pipeline: A valid compiled Pipeline IR proto to be filtered.
    pipeline_run_id_fn: A Callable used for resolving inputs in cases where
      the output of a deleted node is needed by another node that is
      not deleted. The Callable should take a pipeline_pb2.InputSpec.Channel
      message that is to be resolved, and return the pipeline_run_id used to
      resolve it.
    from_nodes: A predicate function that selects nodes by their ids. The set of
      nodes whose node_ids return True determine where the "sweep" starts from
      (see detailed description).
      This defaults to lambda _: True (i.e., select all nodes).
    to_nodes: A predicate function that selects nodes by their ids. The set of
      nodes whose node_ids return True determine where the "sweep" ends (see
      detailed description).
      This defaults to lambda _: True (i.e., select all nodes).
    skip_nodes: A predicate function that selects nodes by their ids. The set of
      nodes whose node_ids return True will not be run, regardless of whether
      they are between from_nodes and to_nodes.
      This defaults to lambda _: False (i.e., do not skip any node).

  Returns:
    The filtered Pipeline IR proto.

  Raises:
    ValueError: If input_pipeline contains a subpipeline.
  """
  if any(
      pipeline_or_node.HasField('sub_pipeline')
      for pipeline_or_node in input_pipeline.nodes):
    raise ValueError('Pipeline filtering not supported for '
                     'pipelines with sub-pipelines.')

  if from_nodes is None:
    from_nodes = lambda _: True
  if to_nodes is None:
    to_nodes = lambda _: True
  if skip_nodes is None:
    skip_nodes = lambda _: False

  node_map = _make_ordered_node_map(input_pipeline)
  from_node_ids = [node_id for node_id in node_map if from_nodes(node_id)]
  to_node_ids = [node_id for node_id in node_map if to_nodes(node_id)]
  skip_node_ids = [node_id for node_id in node_map if skip_nodes(node_id)]
  node_map = _filter_node_map(node_map, from_node_ids, to_node_ids,
                              skip_node_ids)
  node_map = _fix_nodes(node_map, pipeline_run_id_fn)
  fixed_deployment_config = _fix_deployment_config(input_pipeline, node_map)
  return _make_filtered_pipeline(input_pipeline, node_map,
                                 fixed_deployment_config)


def make_latest_resolver_pipeline_run_id_fn(
    metadata_connection_config: mlmd_pb2.ConnectionConfig
) -> Callable[[p_pb2.InputSpec.Channel], str]:
  """Makes a pipeline_run_id_fn that automatically resolves pipeline_run_ids."""
  mlmd_client = mlmd_analytics.Analytics(metadata_connection_config)

  def _pipeline_run_id_fn(channel):
    pipeline_run = mlmd_client.get_latest_pipeline_run(
        component_id=channel.producer_node_query.id)
    return pipeline_run.run_id

  return _pipeline_run_id_fn


class _Direction(enum.Enum):
  UPSTREAM = 1
  DOWNSTREAM = 2


def _make_ordered_node_map(
    pipeline: p_pb2.Pipeline
) -> 'collections.OrderedDict[str, p_pb2.PipelineNode]':
  """Helper function to prepare the Pipeline proto for DAG traversal.

  Args:
    pipeline: The input Pipeline proto. Since we expect this to come from the
      compiler, we assume that it is already topologically sorted.

  Returns:
    An OrderedDict that map node_ids to PipelineNodes.
  """
  node_map = collections.OrderedDict()
  for pipeline_or_node in pipeline.nodes:
    node_id = pipeline_or_node.pipeline_node.node_info.id
    node_map[node_id] = pipeline_or_node.pipeline_node
  return node_map


def _traverse(node_map: Mapping[str, p_pb2.PipelineNode], direction: _Direction,
              start_nodes: Collection[str]) -> Set[str]:
  """Traverse a DAG from start_nodes, either upstream or downstream.

  Args:
    node_map: Mapping of node_id to nodes.
    direction: _Direction.UPSTREAM or _Direction.DOWNSTREAM.
    start_nodes: node_ids to start from.

  Returns:
    Set of node_ids visited by this traversal.
  """
  visited_node_ids = set()
  stack = []
  for start_node in start_nodes:
    # Depth-first traversal
    stack.append(start_node)
    while stack:
      current_node_id = stack.pop()
      if current_node_id in visited_node_ids:
        continue
      visited_node_ids.add(current_node_id)
      if direction == _Direction.UPSTREAM:
        stack.extend(node_map[current_node_id].upstream_nodes)
      elif direction == _Direction.DOWNSTREAM:
        stack.extend(node_map[current_node_id].downstream_nodes)
  return visited_node_ids


def _filter_node_map(
    node_map: 'collections.OrderedDict[str, p_pb2.PipelineNode]',
    from_node_ids: Collection[str], to_node_ids: Collection[str],
    skip_node_ids: Collection[str]
) -> 'collections.OrderedDict[str, p_pb2.PipelineNode]':
  """Returns an OrderedDict with only the nodes we want to include."""
  ancestors_of_to_nodes = _traverse(node_map, _Direction.UPSTREAM, to_node_ids)
  descendents_of_from_nodes = _traverse(node_map, _Direction.DOWNSTREAM,
                                        from_node_ids)
  nodes_to_keep = ancestors_of_to_nodes.intersection(
      descendents_of_from_nodes) - set(skip_node_ids)
  filtered_node_map = collections.OrderedDict()
  for node_id, node in node_map.items():
    if node_id in nodes_to_keep:
      filtered_node_map[node_id] = node
  return filtered_node_map


def _remove_dangling_downstream_nodes(
    node: p_pb2.PipelineNode,
    node_ids_to_keep: Collection[str]) -> p_pb2.PipelineNode:
  """Remove node.downstream_nodes that have been filtered out."""
  # Using a loop instead of set intersection to ensure the same order.
  downstream_nodes_to_keep = [
      downstream_node for downstream_node in node.downstream_nodes
      if downstream_node in node_ids_to_keep
  ]
  if len(downstream_nodes_to_keep) == len(node.downstream_nodes):
    return node
  result = p_pb2.PipelineNode()
  result.CopyFrom(node)
  result.downstream_nodes[:] = downstream_nodes_to_keep
  return result


def _replace_pipeline_run_id_in_channel(channel: p_pb2.InputSpec.Channel,
                                        pipeline_run_id: str):
  """Update in place."""
  for context_query in channel.context_queries:
    if context_query.type.name == dsl_constants.PIPELINE_RUN_CONTEXT_TYPE_NAME:
      context_query.name.field_value.CopyFrom(
          mlmd_pb2.Value(string_value=pipeline_run_id))
      return

  channel.context_queries.append(
      p_pb2.InputSpec.Channel.ContextQuery(
          type=mlmd_pb2.ContextType(
              name=dsl_constants.PIPELINE_RUN_CONTEXT_TYPE_NAME),
          name=p_pb2.Value(
              field_value=mlmd_pb2.Value(string_value=pipeline_run_id))))


def _handle_missing_inputs(
    node: p_pb2.PipelineNode,
    node_ids_to_keep: Collection[str],
    pipeline_run_id_fn: Callable[[p_pb2.InputSpec.Channel], str],
) -> p_pb2.PipelineNode:
  """Private helper function to handle missing inputs.

  Args:
    node: The Pipeline node to check for missing inputs.
    node_ids_to_keep: The node_ids that are not filtered out.
    pipeline_run_id_fn: If this node has upstream nodes that are filtered out,
      this function would be used to obtain the pipeline_run_id for that input
      channel, which would then be provided as the 'pipeline_run_id' in the
      'pipeline_run' ContextQuery.

  Returns:
    A copy of the Pipeline node where all inputs that reference filtered-out
    nodes would have their 'pipeline_run' ContextQuery updated.
  """
  upstream_nodes_to_replace = set()
  upstream_nodes_to_keep = []
  for upstream_node in node.upstream_nodes:
    if upstream_node in node_ids_to_keep:
      upstream_nodes_to_keep.append(upstream_node)
    else:
      upstream_nodes_to_replace.add(upstream_node)

  if not upstream_nodes_to_replace:
    return node  # No parent missing, no need to change anything.

  result = p_pb2.PipelineNode()
  result.CopyFrom(node)
  for input_spec in result.inputs.inputs.values():
    for channel in input_spec.channels:
      if channel.producer_node_query.id in upstream_nodes_to_replace:
        pipeline_run_id = pipeline_run_id_fn(channel)
        _replace_pipeline_run_id_in_channel(channel, pipeline_run_id)
  result.upstream_nodes[:] = upstream_nodes_to_keep
  return result


def _fix_nodes(
    node_map: 'collections.OrderedDict[str, p_pb2.PipelineNode]',
    pipeline_run_id_fn: Callable[[p_pb2.InputSpec.Channel], str],
) -> 'collections.OrderedDict[str, p_pb2.PipelineNode]':
  """Remove dangling references and handle missing inputs."""
  new_node_map = collections.OrderedDict()
  for node_id in node_map:
    new_node = _remove_dangling_downstream_nodes(node_map[node_id], node_map)
    new_node = _handle_missing_inputs(new_node, node_map, pipeline_run_id_fn)
    new_node_map[node_id] = new_node
  return new_node_map


def _fix_deployment_config(
    input_pipeline: p_pb2.Pipeline,
    node_ids_to_keep: Collection[str]) -> Union[any_pb2.Any, None]:
  """Filter per-node deployment configs.

  Cast deployment configs from Any proto to IntermediateDeploymentConfig.
  Take all three per-node fields and filter out the nodes using
  node_ids_to_keep. This works because those fields don't contain references to
  other nodes.

  Args:
    input_pipeline: The input Pipeline IR proto.
    node_ids_to_keep: Set of node_ids to keep.

  Returns:
    If the deployment_config field is set in the input_pipeline, this would
    output the deployment config with filtered per-node configs, then cast into
    an Any proto. If the deployment_config field is unset in the input_pipeline,
    then this function would return None.
  """
  if not input_pipeline.HasField('deployment_config'):
    return None

  deployment_config = p_pb2.IntermediateDeploymentConfig()
  input_pipeline.deployment_config.Unpack(deployment_config)

  def _fix_per_node_config(config_map: MutableMapping[str, Any]):
    # We have to make two passes because we cannot modify the dictionary while
    # iterating over it.
    node_ids_to_delete = [
        node_id for node_id in config_map if node_id not in node_ids_to_keep
    ]
    for node_id_to_delete in node_ids_to_delete:
      del config_map[node_id_to_delete]

  _fix_per_node_config(deployment_config.executor_specs)
  _fix_per_node_config(deployment_config.custom_driver_specs)
  _fix_per_node_config(deployment_config.node_level_platform_configs)

  result = any_pb2.Any()
  result.Pack(deployment_config)
  return result


def _make_filtered_pipeline(
    input_pipeline: p_pb2.Pipeline,
    node_map: 'collections.OrderedDict[str, p_pb2.PipelineNode]',
    fixed_deployment_config: Optional[any_pb2.Any] = None) -> p_pb2.Pipeline:
  """Piece different parts of the Pipeline proto together."""
  result_pipeline = p_pb2.Pipeline()
  result_pipeline.CopyFrom(input_pipeline)
  del result_pipeline.nodes[:]
  result_pipeline.nodes.extend(
      p_pb2.Pipeline.PipelineOrNode(pipeline_node=node_map[node_id])
      for node_id in node_map)
  if fixed_deployment_config:
    result_pipeline.deployment_config.CopyFrom(fixed_deployment_config)
  return result_pipeline
