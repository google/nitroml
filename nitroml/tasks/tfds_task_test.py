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
"""Tests for nitroml.tasks.tfds_task.py."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from nitroml.tasks import tfds_task
import tensorflow_datasets as tfds


class TFDSTaskTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'titanic',
          'dataset_fn': lambda: tfds_task.TFDSTask(tfds.builder('titanic')),
          'want_name': 'titanic'
      }, {
          'testcase_name':
              'fashion_mnist',
          'dataset_fn':
              lambda: tfds_task.TFDSTask(tfds.builder('fashion_mnist')),
          'want_name':
              'fashion_mnist'
      })
  def test_examples(self, dataset_fn, want_name):
    with tfds.testing.mock_data(
        num_examples=5,
        data_dir=os.path.join(os.path.dirname(__file__), 'testdata')):
      dataset = dataset_fn()
      self.assertEqual(want_name, dataset.name)
      self.assertIsNotNone(dataset.examples)
      self.assertLen(dataset.components, 1)


if __name__ == '__main__':
  absltest.main()
