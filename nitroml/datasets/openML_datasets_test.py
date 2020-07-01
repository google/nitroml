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
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from absl.testing import absltest, parameterized

from nitroml.datasets import openML_datasets, dataset


class OpenMLDatasetsTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'openML_default',
          'data_dir': '/tmp/',
          'force_download': False
      }, {
          'testcase_name': 'openML_force_download',
          'data_dir': '/tmp/',
          'force_download': True
      })
  def test_examples(self, data_dir, force_download):
    datasets = openML_datasets.OpenMLDataset(data_dir, force_download)
    self.assertEqual(len(datasets.tasks), len(datasets.components))
    self.assertLen(datasets.components, 72)


if __name__ == '__main__':
  absltest.main()
