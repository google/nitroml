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
"""Tests for nitroml.components.publisher.executor."""

import os
from typing import Text, Dict, List, Any

from absl.testing import absltest
from nitroml.benchmark.result import BenchmarkResult
from nitroml.benchmark.result_publisher.executor import BenchmarkResultPublisherExecutor

from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._base_dir = os.path.dirname(os.path.dirname(__file__))

  def _make_input_dict(self, uri: Text = '') -> Dict[Text, List[Artifact]]:
    evaluation = standard_artifacts.ModelEvaluation()
    evaluation.uri = uri
    return {'evaluation': [evaluation]}

  def _make_output_dict(self) -> Dict[Text, List[Artifact]]:
    return {'benchmark_result': [BenchmarkResult()]}

  def _make_executor_properties(self, name: Text) -> Dict[Text, Any]:
    return {'benchmark_name': name, 'run': 1, 'num_runs': 3,
            'additional_context': '{"test_int": 1}'}

  def _get_eval_metrics(
      self, output_dict: Dict[Text, List[Artifact]]) -> Dict[str, str]:
    benchmark_result = output_dict['benchmark_result'][0]
    benchmark_result = benchmark_result.mlmd_artifact
    return {
        k: v.int_value if v.int_value else v.string_value
        for k, v in benchmark_result.custom_properties.items()
    }

  def testDoWithValidEvaluatorURI(self):
    test_evaluation_uri = os.path.join(
        self._base_dir, 'testdata', 'benchmark_results',
        'Evaluator.AutoTFXTasks.benchmark.openml_vowel', 'evaluation', '1062')
    input_dict = self._make_input_dict(test_evaluation_uri)
    output_dict = self._make_output_dict()
    exec_prop = self._make_executor_properties('test')
    executor = BenchmarkResultPublisherExecutor()

    executor.Do(input_dict, output_dict, exec_prop)

    self.assertEqual(
        {
            'accuracy': '0.042553190141916275',
            'average_loss': '2.397735834121704',
            'post_export_metrics/example_count': '94.0',
            'run': 1,
            'num_runs': 3,
            'benchmark': 'test',
            'test_int': '1',
        }, self._get_eval_metrics(output_dict))

  def testDoWithInvalidEvaluatorURIThrows(self):
    input_dict = self._make_input_dict('/invalid_path')
    output_dict = self._make_output_dict()
    exec_prop = self._make_executor_properties('test')
    executor = BenchmarkResultPublisherExecutor()

    with self.assertRaises(ValueError):
      executor.Do(input_dict, output_dict, exec_prop)


if __name__ == '__main__':
  absltest.main()
