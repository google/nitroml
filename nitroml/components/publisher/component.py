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

from typing import Text

from nitroml.components.publisher import executor
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class BenchmarkResult(Artifact):
  """NitroML's custom Artifact to store benchmark results."""
  TYPE_NAME = 'NitroML.BenchmarkResult'


class BenchmarkResultPublisherSpec(ComponentSpec):
  """BenchmarkResultPublisher component spec."""
  PARAMETERS = {
      'benchmark_name': ExecutionParameter(type=Text),
      'run': ExecutionParameter(type=int),
      'num_runs': ExecutionParameter(type=int),
  }
  INPUTS = {
      'evaluation': ChannelParameter(type=standard_artifacts.ModelEvaluation),
  }
  OUTPUTS = {
      'benchmark_result': ChannelParameter(type=BenchmarkResult),
  }


class BenchmarkResultPublisher(base_component.BaseComponent):
  """Component for publishing NitroML benchmark results to MLMD."""

  SPEC_CLASS = BenchmarkResultPublisherSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor.BenchmarkResultPublisherExecutor)

  def __init__(self, benchmark_name: Text, evaluation: Channel, run: int,
               num_runs: int):
    """Construct a BenchmarkResultPublisher.

    Args:
      benchmark_name: A unique name of a benchmark.
      evaluation: A Channel of `ModelEvaluation` to load evaluation results.
      run: The integer benchmark run repetition.
      num_runs: The integer total number of benchmark run repetitions.
    """
    if not benchmark_name:
      raise ValueError(
          'A valid benchmark name is required to run BenchmarkResultPublisher component.'
      )
    if not evaluation:
      raise ValueError(
          'An evaluation channel is required to run BenchmarkResultPublisher component.'
      )
    if not run:
      raise ValueError(
          'A positive is required to run BenchmarkResultPublisher component.')
    if not num_runs:
      raise ValueError(
          'A positive num_runs is required to run BenchmarkResultPublisher component.'
      )

    benchmark_result = channel_utils.as_channel([BenchmarkResult()])

    spec = BenchmarkResultPublisherSpec(
        benchmark_name=benchmark_name,
        run=run,
        num_runs=num_runs,
        evaluation=evaluation,
        benchmark_result=benchmark_result)

    super(BenchmarkResultPublisher, self).__init__(spec=spec)
