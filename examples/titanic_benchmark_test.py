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

from nitroml import results
from examples import titanic_benchmark
from nitroml.testing import e2etest

from ml_metadata import metadata_store


class TitanicBenchmarkTest(e2etest.TestCase):

  def setUp(self):
    super(TitanicBenchmarkTest, self).setUp("nitroml_titanic_benchmark")

  def test(self):
    self.run_benchmarks([titanic_benchmark.TitanicBenchmark()])

    self.assertComponentExecutionCount(7)
    self.assertComponentSucceeded("ImportExampleGen.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded("SchemaGen.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded("StatisticsGen.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded("Transform.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded("Trainer.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded("Evaluator.TitanicBenchmark.benchmark")
    self.assertComponentSucceeded(
        "BenchmarkResultPublisher.TitanicBenchmark.benchmark")

    # Load benchmark results.
    store = metadata_store.MetadataStore(self.metadata_config)
    df = results.overview(store)

    # Check benchmark results overview values.
    self.assertEqual(len(df.index), 1)
    self.assertContainsSubset([
        "benchmark",
        "run",
        "num_runs",
        "auc",
        "accuracy",
        "average_loss",
        "post_export_metrics/example_count",
    ], df.columns.values.tolist())
    self.assertSameElements([1], df["run"].tolist())
    self.assertSameElements([1], df["num_runs"].tolist())
    self.assertSameElements(["TitanicBenchmark.benchmark"],
                            df.benchmark.unique())


if __name__ == "__main__":
  e2etest.main()
