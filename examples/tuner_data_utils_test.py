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
"""Tests for examples.tuner_data_utils."""

import json
import os

from absl.testing import absltest
from examples import tuner_data_utils as tuner_utils
import numpy as np
import tensorflow as tf


class TunerDataUtils(absltest.TestCase):

  def test_aggregate_tuner_data(self):

    d1 = {
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 3, 7, 6],
    }
    d2 = {
        'a': [2, 1, 2, 0, 4],
        'b': [0, 1, 2, 3, 4],
    }
    agg_data = tuner_utils.aggregate_tuner_data(['a', 'b'], [d1, d2])
    self.assertEqual(list(agg_data['a_mean']), [1.5, 1.5, 2.5, 2., 4.5])
    self.assertEqual(list(agg_data['b_mean']), [1., 2.5, 2.5, 5., 5.])
    np.testing.assert_almost_equal(
        agg_data['a_stdev'],
        [0.35355339, 0.35355339, 0.35355339, 1.41421356, 0.35355339])
    np.testing.assert_almost_equal(
        list(agg_data['b_stdev']),
        [0.70710678, 1.06066017, 0.35355339, 1.41421356, 0.70710678])

  def test_display_tuner_data_with_error_bars(self):

    data_path = os.path.join(
        os.path.dirname(__file__), 'testdata', 'tuner_data.json')
    with tf.io.gfile.GFile(data_path, mode='r') as fin:
      tuner_data = json.load(fin)
    tuner_utils.display_tuner_data_with_error_bars(tuner_data)


if __name__ == '__main__':
  absltest.main()
