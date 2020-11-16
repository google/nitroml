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
"""Tests for nitroml.automl.metalearning.metafeature_gen.executor."""

import json
import os
import tempfile

from absl import flags
from absl.testing import absltest
from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.metafeature_gen import executor
import tensorflow as tf
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    source_data_dir = os.path.dirname(os.path.dirname(__file__))
    input_data_dir = os.path.join(source_data_dir, 'testdata')

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

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                       tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)),
        self._testMethodName)
    self._metafeatures = artifacts.MetaFeatures()
    self._metafeatures.uri = output_data_dir
    self._output_dict = {
        executor.METAFEATURES_KEY: [self._metafeatures],
    }

    self._exec_properties = {
        'custom_config': {
            'problem_statement_path': '/some/fake/path'
        }
    }

  def _verify_metafeature_gen_outputs(self):
    self.assertNotEmpty(tf.io.gfile.listdir(self._metafeatures.uri))
    metafeature_path = os.path.join(self._metafeatures.uri,
                                    artifacts.MetaFeatures.DEFAULT_FILE_NAME)
    metafeature = json.loads(io_utils.read_string_file(metafeature_path))
    self.assertEqual(metafeature['num_examples'], 3)
    self.assertEqual(metafeature['num_int_features'], 1)
    self.assertEqual(metafeature['num_float_features'], 1)
    self.assertEqual(metafeature['num_categorical_features'], 2)

  def test_metafeature_gen_do(self):

    ex = executor.MetaFeatureGenExecutor()
    ex.Do(self._input_dict, self._output_dict, self._exec_properties)
    self._verify_metafeature_gen_outputs()


if __name__ == '__main__':
  absltest.main()
