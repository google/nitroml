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
"""Tests for nitroml.components.metalearning.metalearner.executor."""

import os
import tempfile

from absl import flags
from absl.testing import absltest
from nitroml.components.metalearning import artifacts
from nitroml.components.metalearning.metalearner import executor
import tensorflow as tf
from tfx.types import standard_artifacts


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    source_data_dir = os.path.dirname(os.path.dirname(__file__))
    input_data_dir = os.path.join(source_data_dir, 'testdata')
    meta_train_data = {}
    metadata_indices = [1, 2]
    for ix, dataset_id in enumerate(metadata_indices):
      metafeatures = artifacts.MetaFeatures()
      metafeatures.uri = os.path.join(
          input_data_dir, f'MetaFeatureGen.train_mockdata_{dataset_id}',
          'metafeatures', '1')
      hparams = standard_artifacts.HyperParameters()
      hparams.uri = os.path.join(input_data_dir,
                                 f'Tuner.train_mockdata_{dataset_id}',
                                 'best_hyperparameters', '1')
      meta_train_data[f'meta_train_features_{ix}'] = [metafeatures]
      meta_train_data[f'hparams_train_{ix}'] = [hparams]
    self._input_dict = {
        **meta_train_data,
    }

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                       tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)),
        self._testMethodName)
    self._hparams_out = standard_artifacts.HyperParameters()
    self._hparams_out.uri = os.path.join(output_data_dir, 'hparams_out')
    self._model_out = standard_artifacts.Model()
    self._model_out.uri = os.path.join(output_data_dir, 'model')
    self._output_dict = {
        executor.OUTPUT_HYPERPARAMS: [self._hparams_out],
        executor.OUTPUT_MODEL: [self._model_out],
    }

    # Create exec properties.
    self._exec_properties = {
        'custom_config': {
            'problem_statement_path': '/some/fake/path'
        },
    }

  def _verify_hparams_outputs(self):
    self.assertNotEmpty(tf.io.gfile.listdir(self._hparams_out.uri))

  def test_metalearner_majority_voting(self):
    exec_properties = self._exec_properties
    exec_properties['algorithm'] = executor.MAJORITY_VOTING
    ex = executor.MetaLearnerExecutor()
    ex.Do(self._input_dict, self._output_dict, exec_properties)
    self._verify_hparams_outputs()


if __name__ == '__main__':
  absltest.main()
