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
"""Tests for nitroml.automl.autodata.subpipeline."""
# NOTE: When using --test_strategy=local, this test needs to be run with the
# --notest_loasd flag for Borgbox, i.e.
# $ blaze test --test_strategy=local --notest_loasd :test_target
# See https://yaqs.corp.google.com/eng/q/5892565461237760#a4780535983505408

from typing import List

from nitroml.automl.autodata import subpipeline
from nitroml.automl.autodata.preprocessors import basic_preprocessor
from nitroml.benchmark.tasks import tfds_task
from nitroml.testing import e2etest
import tensorflow_datasets as tfds


class AutoDataTest(e2etest.TestCase):

  @e2etest.parameterized.named_parameters(
      {
          "testcase_name": "default",
          "ignore_features": []
      }, {
          "testcase_name": "ignore_name",
          "ignore_features": ["name"]
      })
  def test(self, ignore_features: List[str]):
    """Runs the AutoData pipeline on a TFDS task."""

    task = tfds_task.TFDSTask(tfds.builder("titanic"))
    autodata = subpipeline.AutoData(
        task.problem_statement,
        examples=task.train_and_eval_examples,
        ignore_features=ignore_features,
        preprocessor=basic_preprocessor.BasicPreprocessor())

    self.run_pipeline(components=task.components + autodata.components)

    self.assertComponentSucceeded("ImportExampleGen")
    self.assertComponentSucceeded("SchemaGen.AutoData")
    if ignore_features:
      self.assertComponentSucceeded("annotate_schema.AutoData")
    self.assertComponentSucceeded("StatisticsGen.AutoData")
    self.assertComponentSucceeded("Transform.AutoData")


if __name__ == "__main__":
  e2etest.main()
