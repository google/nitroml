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
"""Tests for learning.elated_zebra.nitroml.examples.google.basic_benchmark_test."""
# NOTE: When using --test_strategy=local, this test needs to be run with the
# --notest_loasd flag for Borgbox, i.e.
# $ blaze test --test_strategy=local --notest_loasd :test_target
# See https://yaqs.corp.google.com/eng/q/5892565461237760#a4780535983505408

import json
import os
import sys
import tempfile

from absl import flags
from absl.testing import absltest
from nitroml import nitroml
from nitroml import results
from nitroml.examples import titanic_benchmark
from nitroml.testing import e2etest

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS


class TitanicBenchmarkTest(e2etest.TestCase):

  def setUp(self):
    super(TitanicBenchmarkTest, self).setUp()
    flags.FLAGS(sys.argv)

    tempdir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self._pipeline_name = "nitroml_titanic_benchmark"
    self._pipeline_root = os.path.join(tempdir, "nitroml", "titanic",
                                       self._pipeline_name)
    self._metadata_path = os.path.join(tempdir, "nitroml", "titanic",
                                       self._pipeline_name, "metadata.db")

  def test(self):
    metadata_config = metadata_store_pb2.ConnectionConfig(
        sqlite=metadata_store_pb2.SqliteMetadataSourceConfig(
            filename_uri=self._metadata_path))
    nitroml.run([titanic_benchmark.TitanicBenchmark()],
                pipeline_name=self._pipeline_name,
                pipeline_root=self._pipeline_root,
                metadata_connection_config=metadata_config)

    self.assertComponentExecutionCount(7, self._metadata_path)
    self.assertComponentSucceeded("ImportExampleGen.TitanicBenchmark.benchmark",
                                  self._metadata_path)
    self.assertComponentSucceeded("SchemaGen.TitanicBenchmark.benchmark",
                                  self._metadata_path)
    self.assertComponentSucceeded("StatisticsGen.TitanicBenchmark.benchmark",
                                  self._metadata_path)
    self.assertComponentSucceeded("Transform.TitanicBenchmark.benchmark",
                                  self._metadata_path)
    self.assertComponentSucceeded("Trainer.TitanicBenchmark.benchmark",
                                  self._metadata_path)
    self.assertComponentSucceeded("Evaluator.TitanicBenchmark.benchmark",
                                  self._metadata_path)
    self.assertComponentSucceeded(
        "BenchmarkResultPublisher.TitanicBenchmark.benchmark",
        self._metadata_path)

    # Load benchmark results.
    store = metadata_store.MetadataStore(metadata_config)
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
  absltest.main()
