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
"""Tests for nitroml.examples.titanic_benchmark."""

from nitroml.benchmark import results
from examples import titanic_benchmark
from nitroml.testing import e2etest

from ml_metadata import metadata_store


class TitanicBenchmarkTest(e2etest.BenchmarkTestCase):

  @e2etest.parameterized.named_parameters(
      {
          "testcase_name": "keras_trainer",
          "use_keras": True,
          "enable_tuning": False,
      }, {
          "testcase_name": "keras_tuning_trainer",
          "use_keras": True,
          "enable_tuning": True,
      }, {
          "testcase_name": "estimator_trainer",
          "use_keras": False,
          "enable_tuning": False,
      })
  def test(self, use_keras, enable_tuning):
    self.run_benchmarks([titanic_benchmark.TitanicBenchmark()],
                        use_keras=use_keras,
                        enable_tuning=enable_tuning)
    if enable_tuning:
      self.assertComponentSucceeded(
          "Tuner.AutoTrainer.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded("ImportExampleGen.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded(
        "SchemaGen.AutoData.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded(
        "StatisticsGen.AutoData.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded(
        "Transform.AutoData.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded(
        "Trainer.AutoTrainer.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded("Evaluator.model.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded(
        "BenchmarkResultPublisher.model.TitanicBenchmark.benchmark")

    # Load benchmark results.
    store = metadata_store.MetadataStore(self.metadata_config)
    df = results.overview(store)

    # Check benchmark results overview values.
    self.assertEqual(len(df.index), 1)
    self.assertContainsSubset([
        "run_id",
        "benchmark_fullname",
        "benchmark",
        "run",
        "num_runs",
        "accuracy",
        "auc_pr",
        "auc_roc",
        "example_count",
        "precision",
        "recall",
        "weighted_example_count",
    ], df.columns.values.tolist())
    self.assertSameElements([1], df["run"].tolist())
    self.assertSameElements([1], df["num_runs"].tolist())
    self.assertSameElements(["TitanicBenchmark.benchmark"],
                            df.benchmark.unique())


if __name__ == "__main__":
  e2etest.main()
