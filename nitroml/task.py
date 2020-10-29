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
r"""A model quality benchmark task."""

import abc
from typing import List, Optional

from nitroml.components.publisher.component import BenchmarkResultPublisher
import tensorflow_model_analysis as tfma
from tfx import components as tfx
from tfx import types
from tfx.dsl.components.base import base_component

from nitroml.protos import problem_statement_pb2 as ps_pb2


class BenchmarkTask(abc.ABC):
  r"""Defines a BenchmarkTask for AutoML."""

  @abc.abstractproperty
  def name(self) -> str:
    """Returns the task's name."""

  @abc.abstractproperty
  def components(self) -> List[base_component.BaseComponent]:
    """Returns TFX components required for the task."""

  @abc.abstractproperty
  def train_and_eval_examples(self) -> types.Channel:
    """Returns train and eval labeled examples."""

  @abc.abstractproperty
  def problem_statement(self) -> ps_pb2.ProblemStatement:
    """Returns the ProblemStatement associated with this BenchmarkTask."""

  def make_evaluation(
      self,
      model: types.Channel,
      benchmark_name: str,
      benchmark_run: int,
      runs_per_benchmark: int,
      eval_config: Optional[tfma.EvalConfig] = None,
      **kwargs,
  ) -> List[base_component.BaseComponent]:
    """Returns a list of components for evaluating the model.

    Evaluates the model on the 'eval' split using the TFX Evaluator, and
    publishes results to MLMD for analysis from the results Colab notebook.

    TODO(b/171835684): Don't require passing `benchmark_run` and
    `runs_per_benchmark`.

    Args:
      model: A `standard_artifacts.Model` Channel, usually produced by a Trainer
        component. Input to the benchmark Evaluator.
      benchmark_name: Unique name of the benchmark, used for publishing
        evalution results.
      benchmark_run: Index of the benchmark run for when runs_per_benchmark > 1.
      runs_per_benchmark: Integer number of runs_per_benchmark. Used for
        computing means and variance for benchmark runs.
      eval_config: A TFMA `EvalConfig` for customizing the TFMA evaluation.
        Required when `model` was produced from `tf.keras.Model#save`.
      **kwargs: Additional kwargs to pass to `Task#make_evaluation`.
    """

    del kwargs  # Unused.

    evaluator = tfx.Evaluator(
        examples=self.train_and_eval_examples,
        model=model,
        eval_config=eval_config,
        example_splits=['eval'])

    # TODO(weill): Remove `benchmark_name` once name_scope is supported in TFX.
    publisher = BenchmarkResultPublisher(
        benchmark_name,
        evaluator.outputs.evaluation,
        run=benchmark_run,
        num_runs=runs_per_benchmark)

    return [evaluator, publisher]
