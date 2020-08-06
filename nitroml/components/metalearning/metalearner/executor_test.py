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

import json
import os
import tempfile

from absl import flags
from absl import logging
from absl.testing import absltest
from nitroml.components.metalearning import artifacts
from nitroml.components.metalearning.metalearner import executor
import tensorflow as tf
from tfx.utils import io_utils
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    source_data_dir = os.path.dirname(os.path.dirname(__file__))
    self._input_data_dir = os.path.join(source_data_dir, 'testdata')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                       tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)),
        self._testMethodName)
    self._hparams_out = standard_artifacts.HyperParameters()
    self._hparams_out.uri = os.path.join(output_data_dir, 'hparams_out')
    self._model_out = standard_artifacts.Model()
    self._model_out.uri = os.path.join(output_data_dir, 'model')

    # Create exec properties.
    self._exec_properties = {
        'custom_config': {
            'problem_statement_path': '/some/fake/path'
        },
    }

  def _verify_hparams_outputs(self):

    path = os.path.join(self._hparams_out.uri, 'meta_hyperparameters.txt')
    self.assertTrue(tf.io.gfile.exists(path))
    hparams_json = json.loads(io_utils.read_string_file(path))
    search_space = hparams_json['space']
    for hspace in search_space:
      hspace = hspace['config']
      self.assertIn(hspace['name'],
                    ['learning_rate', 'optimizer', 'num_layers', 'num_nodes'])

      if hspace['name'] == 'learning_rate':
        self.assertEqual(hspace['values'], [0.01, 0.1, 0.001])
      elif hspace['name'] == 'optimizer':
        self.assertEqual(hspace['values'], ['RMSprop'])
      elif hspace['name'] == 'num_layers':
        self.assertEqual(hspace['values'], [4])
      elif hspace['name'] == 'num_nodes':
        self.assertEqual(hspace['values'], [128, 64, 32])

  def test_metalearner_majority_voting(self):

    meta_train_data = {}
    metadata_indices = [1, 2, 3]
    for ix, dataset_id in enumerate(metadata_indices):
      hparams = standard_artifacts.HyperParameters()
      hparams.uri = os.path.join(self._input_data_dir,
                                 f'Tuner.train_mockdata_{dataset_id}',
                                 'best_hyperparameters')
      meta_train_data[f'hparams_train_{ix}'] = [hparams]

    input_dict = {
        **meta_train_data,
    }
    output_dict = {
        executor.OUTPUT_HYPERPARAMS: [self._hparams_out],
        executor.OUTPUT_MODEL: [self._model_out],
    }

    exec_properties = self._exec_properties.copy()
    exec_properties['algorithm'] = executor.MAJORITY_VOTING
    ex = executor.MetaLearnerExecutor()
    ex.Do(input_dict, output_dict, exec_properties)

    self._verify_hparams_outputs()


if __name__ == '__main__':
  absltest.main()
