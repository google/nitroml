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
"""Tests for nitroml.components.meta_learning.meta_feature_gen.executor."""

import os
import tempfile

from absl import flags
from absl import logging
from absl.testing import absltest
from nitroml.components.meta_learning.meta_feature_gen import executor
from nitroml.components.meta_learning import artifacts
from tfx import types
import tensorflow as tf
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    source_data_dir = os.path.dirname(os.path.dirname(__file__))
    input_data_dir = os.path.join(source_data_dir, 'testdata',
                                  'meta_feature_gen')
    statistics = standard_artifacts.ExampleStatistics()
    statistics.uri = os.path.join(input_data_dir,
                                  'StatisticsGen.train_mockdata_1',
                                  'statistics', '5')
    statistics.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    transformed_examples = standard_artifacts.Examples()
    transformed_examples.uri = os.path.join(input_data_dir,
                                            'Transform.train_mockdata_1',
                                            'transformed_examples', '10')
    transformed_examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    self._input_dict = {
        executor.EXAMPLES_KEY: [transformed_examples],
        executor.STATISTICS_KEY: [statistics],
    }

    # Create output_dict.
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                       tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)),
        self._testMethodName)

    self._meta_features = artifacts.MetaFeatures()
    self._meta_features.uri = output_data_dir
    self._output_dict = {
        executor.META_FEATURES_KEY: [self._meta_features],
    }

    # Create exec properties.
    self._exec_properties = {
        'custom_config': {
            'problem_statement_path': '/some/fake/path'
        }
    }

  def _verify_meta_feature_gen_outputs(self):
    self.assertNotEmpty(tf.io.gfile.listdir(self._meta_features.uri))

  def test_meta_feature_gen_do(self):

    exec = executor.MetaFeatureGenExecutor()
    exec.Do(self._input_dict, self._output_dict, self._exec_properties)
    self._verify_meta_feature_gen_outputs()


if __name__ == '__main__':
  absltest.main()