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
"""Component for publishing NitroML benchmark results to MLMD."""

from typing import Optional

from nitroml.benchmark import result
from nitroml.benchmark.result_publisher import executor
from nitroml.benchmark.result_publisher import serialize
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.channel import Channel
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class BenchmarkResultPublisherSpec(ComponentSpec):
  """BenchmarkResultPublisher component spec."""
  PARAMETERS = {
      'benchmark_name': ExecutionParameter(type=str),
      'run': ExecutionParameter(type=int),
      'num_runs': ExecutionParameter(type=int),
      'additional_context': ExecutionParameter(type=str),
  }
  INPUTS = {
      'evaluation': ChannelParameter(type=standard_artifacts.ModelEvaluation),
  }
  OUTPUTS = {
      'benchmark_result': ChannelParameter(type=result.BenchmarkResult),
  }


class BenchmarkResultPublisher(base_component.BaseComponent):
  """Component for publishing NitroML benchmark results to MLMD."""

  SPEC_CLASS = BenchmarkResultPublisherSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor.BenchmarkResultPublisherExecutor)

  def __init__(self,
               benchmark_name: str,
               evaluation: Channel,
               run: int,
               num_runs: int,
               instance_name: Optional[str] = None,
               additional_context: Optional[serialize.ContextDict] = None):
    """Construct a BenchmarkResultPublisher.

    Args:
      benchmark_name: A unique name of a benchmark.
      evaluation: A Channel of `ModelEvaluation` to load evaluation results.
      run: The integer benchmark run repetition.
      num_runs: The integer total number of benchmark run repetitions.
      instance_name: Optional unique instance name. Necessary iff multiple
        BenchmarkResultPublisher components are declared in the same pipeline.
      additional_context: Additional key-value pairs to be written to
        BenchmarkResultPublisher artifact, used for adding additional context
        to benchmark result, such as dataset names, hyper param values etc.
        Keys must be type str and Values must be primitive types int, str, bool,
        or float.
    """

    if not benchmark_name:
      raise ValueError(
          'A valid benchmark name is required to run BenchmarkResultPublisher component.'
      )
    if not evaluation:
      raise ValueError(
          'An evaluation channel is required to run BenchmarkResultPublisher component.'
      )

    benchmark_result = channel_utils.as_channel([result.BenchmarkResult()])

    if additional_context:
      serialized_additional_context = serialize.encode(additional_context)
    else:
      serialized_additional_context = serialize.encode({})

    spec = BenchmarkResultPublisherSpec(
        benchmark_name=benchmark_name,
        run=run,
        num_runs=num_runs,
        evaluation=evaluation,
        benchmark_result=benchmark_result,
        additional_context=serialized_additional_context)

    super(BenchmarkResultPublisher, self).__init__(
        spec=spec, instance_name=instance_name)
