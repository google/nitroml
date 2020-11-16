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
"""Tests for nitroml.suites.openml_cc18."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from nitroml.benchmark.suites import openml_cc18
from nitroml.benchmark.suites import testing_utils
import requests_mock


class OpenMLCC18Test(parameterized.TestCase, absltest.TestCase):
  """Test cases for datasets.openML_datasets provider.

  There are two test cases:
  1) Test Case 1: Downloads mock data for openML CC18 datasets.
  2) Test Case 2: Checks the cache, if data exists loads from the disk, else
  downloads.
  """

  @parameterized.named_parameters(
      {
          'testcase_name': 'openML_default',
          'use_cache': False,
      }, {
          'testcase_name': 'openML_use-cache',
          'use_cache': True,
      })
  def test_examples(self, use_cache):

    root_dir = os.path.join(absltest.get_default_test_tmpdir(),
                            'openML_mock_data')
    with requests_mock.Mocker() as mocker:
      testing_utils.register_mock_urls(mocker)
      suite = openml_cc18.OpenMLCC18(root_dir, use_cache, mock_data=True)

    self.assertNotEmpty(list(suite))


if __name__ == '__main__':
  absltest.main()
