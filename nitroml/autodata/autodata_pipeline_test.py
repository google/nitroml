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
"""Tests for nitroml.autodata.autodata_pipeline."""
# NOTE: When using --test_strategy=local, this test needs to be run with the
# --notest_loasd flag for Borgbox, i.e.
# $ blaze test --test_strategy=local --notest_loasd :test_target
# See https://yaqs.corp.google.com/eng/q/5892565461237760#a4780535983505408

import os

from nitroml.autodata import autodata_pipeline
from nitroml.autodata.preprocessors import basic_preprocessor
from nitroml.suites import openml_cc18
from nitroml.suites import testing_utils
from nitroml.testing import e2etest
import requests_mock


class AutoDataTest(e2etest.TestCase):

  def setUp(self):
    super(AutoDataTest, self).setUp(pipeline_name="autodata_test")

  def test(self):
    """Runs the AutoData pipeline on a TFDS task."""
    with requests_mock.Mocker() as mocker:
      testing_utils.register_mock_urls(mocker)
      task = list(
          openml_cc18.OpenMLCC18(
              root_dir=os.path.join(self.pipeline_root, "openML_mock_data"),
              mock_data=True))[0]

      autodata = autodata_pipeline.AutoData(
          task.problem_statement,
          examples=task.train_and_eval_examples,
          preprocessor=basic_preprocessor.BasicPreprocessor())

      self.run_pipeline(components=task.components + autodata.components)

      self.assertComponentExecutionCount(4)
      self.assertComponentSucceeded("CsvExampleGen.OpenML.mockdata_1")
      self.assertComponentSucceeded("SchemaGen.AutoData")
      self.assertComponentSucceeded("StatisticsGen.AutoData")
      self.assertComponentSucceeded("Transform.AutoData")


if __name__ == "__main__":
  e2etest.main()
