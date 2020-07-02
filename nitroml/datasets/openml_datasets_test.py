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
"""Tests for nitroml.datasets.openML_datasets.py."""

import tempfile

from absl.testing import absltest, parameterized

from nitroml.datasets import openml_datasets


class OpenMLDatasetsTest(parameterized.TestCase, absltest.TestCase):
  """Test cases for datasets.openML_datasets provider.

  There are two test cases:
  1) Test Case 1: Downloads all 72 openML CC18 datasets. Running this test case may take a while.
  2) Test Case 2: Checks the cache, if data exists loads from the disk, else downloads all the datasets.
  """

  @parameterized.named_parameters(
      {
          'testcase_name': 'openML_default',
          'data_dir': tempfile.gettempdir(),
          'use_cache': False
      }, {
          'testcase_name': 'openML_use-cache',
          'data_dir': tempfile.gettempdir(),
          'use_cache': True
      })
  def test_examples(self, data_dir, use_cache):
    datasets = openml_datasets.OpenMLCC18(data_dir, use_cache)
    self.assertEqual(len(datasets.tasks), len(datasets.components))
    self.assertLen(datasets.components, 72)


if __name__ == '__main__':
  absltest.main()
