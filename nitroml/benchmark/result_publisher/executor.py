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
"""Executor for BenchmarkResultPublisher."""

from typing import Any, Dict, List, Text

from nitroml.benchmark import result as br
from nitroml.benchmark.result_publisher import serialize
import tensorflow.compat.v2 as tf
import tensorflow_model_analysis as tfma

from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact


class BenchmarkResultPublisherExecutor(base_executor.BaseExecutor):
  """Executor for BenchamarkResultPublisher."""

  def Do(self, input_dict: Dict[Text, List[Artifact]],
         output_dict: Dict[Text, List[Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Take evaluator output and publish results to MLMD.

    It updates custom properties of BenchmarkResult artifact to contain
    benchmark results.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - evaluation: Model evaluation results.
      output_dict: Output dict from key to a list of artifacts, including:
        - benchmark_result: `BenchmarkResult` artifact.
      exec_properties: A dict of execution properties, including either one of:
        - benchmark_name: An unique name of a benchmark.

    Raises:
      ValueError: If evaluation uri doesn't exists.
    """

    uri = artifact_utils.get_single_uri(input_dict['evaluation'])
    if not tf.io.gfile.exists(uri):
      raise ValueError('The uri="{}" does not exist.'.format(uri))

    benchmark_result = artifact_utils.get_single_instance(
        output_dict['benchmark_result'])
    benchmark_result.set_string_custom_property(
        br.BenchmarkResult.BENCHMARK_NAME_KEY,
        exec_properties['benchmark_name'])
    benchmark_result.set_int_custom_property(
        br.BenchmarkResult.BENCHMARK_RUN_KEY, exec_properties['run'])
    benchmark_result.set_int_custom_property(
        br.BenchmarkResult.RUNS_PER_BENCHMARK_KEY, exec_properties['num_runs'])

    # Publish evaluation metrics
    evals = self._load_evaluation(uri)
    for name, val in evals.items():
      # TODO(b/151723291): Use correct type instead of string.
      benchmark_result.set_string_custom_property(name, str(val))

    context_properties = serialize.decode(exec_properties['additional_context'])
    # TODO(b/175802446): Add additional properties storing
    # `additional_context` and `metric` keys so user can distinguish between
    # custom properties.

    for name, val in context_properties.items():
      # TODO(b/151723291): Use correct type instead of string.
      benchmark_result.set_string_custom_property(name, str(val))

  def _load_evaluation(self, file_path: Text) -> Dict[str, Any]:
    """Returns evaluations for a bechmark run.

    This method makes following assumptions:
      1. `tf.enable_v2_behavior()` was called beforehand.
      2. file_path points to a dir containing artifacts of single output model.

    Args:
      file_path: A root directory where pipeline's evaluation artifacts are
        stored.

    Returns:
      An evaluation metrics dictionary. If no evaluations found then returns an
      empty dictionary.
    """
    # We assume this is a single output model, hence the following keys are "".
    output_name = ''
    multi_class_key = ''

    eval_result = tfma.load_eval_result(file_path)

    # Slicing_metric is a tuple, index 0 is slice, index 1 is its value.
    _, metrics_dict = eval_result.slicing_metrics[0]

    if output_name not in metrics_dict or multi_class_key not in metrics_dict[
        output_name]:
      raise ValueError('Evaluation can only be loaded for single output model.')
    metrics_dict = metrics_dict[output_name][multi_class_key]

    return {k: v.get('doubleValue') for k, v in metrics_dict.items()}
