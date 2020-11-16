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
"""Tests for nitroml.automl.metalearning.metalearner.executor."""

import json
import os
import tempfile

from absl import flags
from absl.testing import absltest
from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.metalearner import executor
import numpy as np
import tensorflow as tf
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import path_utils


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    source_data_dir = os.path.dirname(os.path.dirname(__file__))
    self._input_data_dir = os.path.join(source_data_dir, 'testdata')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                       tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)),
        self._testMethodName)
    self._hparams_out = artifacts.KCandidateHyperParameters()
    self._hparams_out.uri = os.path.join(output_data_dir, 'hparams_out')
    self._model_out = standard_artifacts.Model()
    self._model_out.uri = os.path.join(output_data_dir, 'model')

    # Create exec properties.
    self._exec_properties = {
        'custom_config': {
            'problem_statement_path': '/some/fake/path'
        },
    }

  def _verify_hparams_values_majority_voting(self, hparams_json_list):

    search_space = hparams_json_list[0]['space']
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

  def _verify_hparams_values_nearest_neighbor(self, hparams_json_list):

    for hparams in hparams_json_list:
      search_space = hparams['space']
      for hspace in search_space:
        hspace = hspace['config']
        self.assertIn(hspace['name'],
                      ['learning_rate', 'optimizer', 'num_layers', 'num_nodes'])

        if hspace['name'] == 'learning_rate':
          self.assertIn(hspace['values'][0], [0.01, 0.1, 0.001])
        elif hspace['name'] == 'optimizer':
          self.assertIn(hspace['values'][0], ['SGD', 'RMSprop'])
        elif hspace['name'] == 'num_layers':
          self.assertIn(hspace['values'][0], [2, 4])
        elif hspace['name'] == 'num_nodes':
          self.assertIn(hspace['values'][0], [128, 64, 32])

  def _verify_hparams_outputs(self, algorithm: str):

    path = os.path.join(self._hparams_out.uri, 'meta_hyperparameters.txt')
    self.assertTrue(tf.io.gfile.exists(path))
    hparams_json_list = json.loads(io_utils.read_string_file(path))

    if algorithm == executor.MAJORITY_VOTING:
      self._verify_hparams_values_majority_voting(hparams_json_list)
    elif algorithm == executor.NEAREST_NEIGHBOR:
      self._verify_hparams_values_nearest_neighbor(hparams_json_list)

  def _verify_model_export(self):
    self.assertTrue(
        tf.io.gfile.exists(path_utils.serving_model_dir(self._model_out.uri)))

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

    self._verify_hparams_outputs(executor.MAJORITY_VOTING)

  def test_metalearner_nearest_neighbor(self):

    meta_train_data = {}
    metadata_indices = [1, 2, 3]
    for ix, dataset_id in enumerate(metadata_indices):
      hparams = standard_artifacts.HyperParameters()
      hparams.uri = os.path.join(self._input_data_dir,
                                 f'Tuner.train_mockdata_{dataset_id}',
                                 'best_hyperparameters')
      meta_train_data[f'hparams_train_{ix}'] = [hparams]

      metafeature = artifacts.MetaFeatures()
      metafeature.uri = os.path.join(
          self._input_data_dir, f'MetaFeatureGen.train_mockdata_{dataset_id}',
          'metafeatures')
      meta_train_data[f'meta_train_features_{ix}'] = [metafeature]

    input_dict = {
        **meta_train_data,
    }
    output_dict = {
        executor.OUTPUT_HYPERPARAMS: [self._hparams_out],
        executor.OUTPUT_MODEL: [self._model_out],
    }

    exec_properties = self._exec_properties.copy()
    exec_properties['algorithm'] = executor.NEAREST_NEIGHBOR
    ex = executor.MetaLearnerExecutor()
    ex.Do(input_dict, output_dict, exec_properties)

    self._verify_hparams_outputs(executor.NEAREST_NEIGHBOR)
    self._verify_model_export()

  def test_create_search_space_using_voting(self):

    metadata_indices = [1, 2, 3]
    all_hparams = []
    for dataset_id in metadata_indices:
      hyperparameters_file = os.path.join(self._input_data_dir,
                                          f'Tuner.train_mockdata_{dataset_id}',
                                          'best_hyperparameters',
                                          'best_hyperparameters.txt')
      hparams_json = json.loads(io_utils.read_string_file(hyperparameters_file))
      all_hparams.append(hparams_json['values'])
    ex = executor.MetaLearnerExecutor()
    search_space = ex._create_search_space_using_voting(all_hparams)
    self._verify_hparams_values_majority_voting([search_space.get_config()])

  def _verify_model_weights(self, model_weights):

    expected_weights = np.array([[0.3487429, 0.81373346, 0.46499056],
                                 [0.57735026, 0.57735026, 0.57735026],
                                 [0.23570228, 0.23570228, 0.9428091],
                                 [0.17609018, 0.17609018, 0.968496]])

    np.testing.assert_array_almost_equal(
        expected_weights, model_weights, decimal=4)

  def test_create_knn_model_from_metafeatures(self):

    metadata_indices = [1, 2, 3]
    metafeatures_list = []
    for dataset_id in metadata_indices:
      metafeature_uri = os.path.join(
          self._input_data_dir, f'MetaFeatureGen.train_mockdata_{dataset_id}',
          'metafeatures', 'metafeatures.txt')
      metafeatures = json.loads(io_utils.read_string_file(metafeature_uri))
      metafeatures_list.append(metafeatures['metafeature'])

    ex = executor.MetaLearnerExecutor()
    model = ex._create_knn_model_from_metafeatures(metafeatures_list)
    model_weights = model.get_layer('metafeatures').get_weights()
    self._verify_model_weights(model_weights[0])


if __name__ == '__main__':
  absltest.main()
