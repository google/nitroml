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
"""Tests for nitroml.components.tuner.executor."""

import json
import os
import tempfile

from absl import flags
from absl.testing import absltest
from kerastuner.engine.hyperparameters import HyperParameters
from nitroml.automl.autotrainer.lib import auto_trainer as tuner_module
from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.tuner import component as tuner_component
from nitroml.automl.metalearning.tuner import executor
import tensorflow as tf
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils

from google.protobuf import json_format
from google.protobuf import text_format
from nitroml.protos import problem_statement_pb2 as ps_pb2


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    self._testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')

    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                       tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)),
        self._testMethodName)

    self._context = executor.Executor.Context(
        tmp_dir=self._output_data_dir, unique_id='1')

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(self._testdata_dir, 'Transform.mockdata_1',
                                'transformed_examples', '10')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._testdata_dir, 'SchemaGen.mockdata_1',
                              'schema', '1')

    transform_output = standard_artifacts.TransformGraph()
    transform_output.uri = os.path.join(self._testdata_dir,
                                        'Transform.mockdata_1',
                                        'transform_graph', '10')

    self._input_dict = {
        'examples': [examples],
        'schema': [schema],
        'transform_graph': [transform_output],
    }

    # Create output dict.
    self._best_hparams = standard_artifacts.Model()
    self._best_hparams.uri = os.path.join(self._output_data_dir, 'best_hparams')

    self._tuner_data = tuner_component.TunerData()
    self._tuner_data.uri = os.path.join(self._output_data_dir,
                                        'trial_summary_plot')
    self._output_dict = {
        'best_hyperparameters': [self._best_hparams],
        'trial_summary_plot': [self._tuner_data],
    }

    # Create exec properties.
    self._exec_properties = {
        'train_args':
            json_format.MessageToJson(
                trainer_pb2.TrainArgs(num_steps=2),
                preserving_proto_field_name=True),
        'eval_args':
            json_format.MessageToJson(
                trainer_pb2.EvalArgs(num_steps=1),
                preserving_proto_field_name=True),
    }

  def _verify_output(self):
    # Test best hparams.
    best_hparams_path = os.path.join(self._best_hparams.uri,
                                     'best_hyperparameters.txt')
    self.assertTrue(tf.io.gfile.exists(best_hparams_path))
    with tf.io.gfile.GFile(best_hparams_path, mode='r') as f:
      best_hparams_config = json.loads(f.read())
    best_hparams = HyperParameters.from_config(best_hparams_config)
    self.assertBetween(best_hparams.get('learning_rate'), 1e-4, 1.01)
    self.assertBetween(best_hparams.get('num_layers'), 1, 5)

  def testDoWithTunerFn(self):

    self._exec_properties['tuner_fn'] = '%s.%s' % (
        tuner_module.tuner_fn.__module__, tuner_module.tuner_fn.__name__)

    ps_type = ps_pb2.Type(
        binary_classification=ps_pb2.BinaryClassification(label='class'))
    ps = ps_pb2.ProblemStatement(
        owner=['nitroml'],
        tasks=[ps_pb2.Task(
            name='mockdata_1',
            type=ps_type,
        )])

    self._exec_properties['custom_config'] = json_utils.dumps({
        'problem_statement':
            text_format.MessageToString(message=ps, as_utf8=True),
    })

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=self._input_dict,
        output_dict=self._output_dict,
        exec_properties=self._exec_properties)

    self._verify_output()

  def testDoWithMajoritVoting(self):

    exec_properties = self._exec_properties.copy()
    exec_properties['tuner_fn'] = '%s.%s' % (tuner_module.tuner_fn.__module__,
                                             tuner_module.tuner_fn.__name__)
    exec_properties['metalearning_algorithm'] = 'majority_voting'

    input_dict = self._input_dict.copy()

    ps_type = ps_pb2.Type(
        binary_classification=ps_pb2.BinaryClassification(label='class'))
    ps = ps_pb2.ProblemStatement(
        owner=['nitroml'],
        tasks=[ps_pb2.Task(
            name='mockdata_1',
            type=ps_type,
        )])

    exec_properties['custom_config'] = json_utils.dumps({
        'problem_statement':
            text_format.MessageToString(message=ps, as_utf8=True),
    })
    hps_artifact = artifacts.KCandidateHyperParameters()
    hps_artifact.uri = os.path.join(self._testdata_dir,
                                    'MetaLearner.majority_voting',
                                    'hparams_out')
    input_dict['warmup_hyperparameters'] = [hps_artifact]

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=input_dict,
        output_dict=self._output_dict,
        exec_properties=exec_properties)
    self._verify_output()

  def testDoWithNearestNeighbor(self):

    exec_properties = self._exec_properties.copy()
    exec_properties['tuner_fn'] = '%s.%s' % (tuner_module.tuner_fn.__module__,
                                             tuner_module.tuner_fn.__name__)
    exec_properties['metalearning_algorithm'] = 'nearest_neighbor'

    input_dict = self._input_dict.copy()

    ps_type = ps_pb2.Type(
        binary_classification=ps_pb2.BinaryClassification(label='class'))
    ps = ps_pb2.ProblemStatement(
        owner=['nitroml'],
        tasks=[ps_pb2.Task(
            name='mockdata_1',
            type=ps_type,
        )])

    exec_properties['custom_config'] = json_utils.dumps({
        'problem_statement':
            text_format.MessageToString(message=ps, as_utf8=True),
    })

    hps_artifact = artifacts.KCandidateHyperParameters()
    hps_artifact.uri = os.path.join(self._testdata_dir,
                                    'MetaLearner.nearest_neighbor',
                                    'hparams_out')
    input_dict['warmup_hyperparameters'] = [hps_artifact]

    model_artifact = standard_artifacts.Model()
    model_artifact.uri = os.path.join(self._testdata_dir,
                                      'MetaLearner.nearest_neighbor', 'model')
    input_dict['metamodel'] = [model_artifact]

    metafeature = artifacts.MetaFeatures()
    metafeature.uri = os.path.join(self._testdata_dir,
                                   'MetaFeatureGen.test_mockdata',
                                   'metafeatures')
    input_dict['metafeature'] = [metafeature]

    tuner = executor.Executor(self._context)
    tuner.Do(
        input_dict=input_dict,
        output_dict=self._output_dict,
        exec_properties=exec_properties)
    self._verify_output()

  def testTuneArgs(self):
    with self.assertRaises(ValueError):
      self._exec_properties['tune_args'] = json_format.MessageToJson(
          tuner_pb2.TuneArgs(num_parallel_trials=3),
          preserving_proto_field_name=True)

      tuner = executor.Executor(self._context)
      tuner.Do(
          input_dict=self._input_dict,
          output_dict=self._output_dict,
          exec_properties=self._exec_properties)

  def testMergeTunerDataFnWithMaxTunerObjective(self):

    data1 = {
        executor.BEST_CUMULATIVE_SCORE: [1, 2, 3.5],
        executor.OBJECTIVE_DIRECTION: 'max',
    }
    data2 = {
        executor.BEST_CUMULATIVE_SCORE: [2.5, 2.8, 3.5, 4.0],
        executor.OBJECTIVE_DIRECTION: 'max',
    }
    (cumulative_data, best_tuner_ix) = executor.merge_trial_data(data1, data2)

    expected_output = [1, 2, 3.5, 3.5, 3.5, 3.5, 4.0]
    self.assertEqual(cumulative_data[executor.BEST_CUMULATIVE_SCORE],
                     expected_output)
    self.assertEqual(best_tuner_ix, 1)
    self.assertEqual(cumulative_data[executor.OBJECTIVE_DIRECTION], 'max')

    data1[executor.BEST_CUMULATIVE_SCORE] = [1, 2, 4.5]
    data2[executor.BEST_CUMULATIVE_SCORE] = [0.5, 0.8, 2.5, 4.0]
    (cumulative_data, best_tuner_ix) = executor.merge_trial_data(data1, data2)
    expected_output = [1, 2, 4.5, 4.5, 4.5, 4.5, 4.5]
    self.assertEqual(cumulative_data[executor.BEST_CUMULATIVE_SCORE],
                     expected_output)
    self.assertEqual(best_tuner_ix, 0)
    self.assertEqual(cumulative_data[executor.OBJECTIVE_DIRECTION], 'max')

  def testMergeTunerDataFnWithMinTunerObjective(self):

    data1 = {
        executor.BEST_CUMULATIVE_SCORE: [3.5, 2, 0.0],
        executor.OBJECTIVE_DIRECTION: 'min',
    }
    data2 = {
        executor.BEST_CUMULATIVE_SCORE: [4.0, 3.5, 2.0, 1],
        executor.OBJECTIVE_DIRECTION: 'min',
    }
    (cumulative_data, best_tuner_ix) = executor.merge_trial_data(data1, data2)
    expected_output = [3.5, 2, 0.0, 0.0, 0.0, 0.0, 0.0]

    self.assertEqual(cumulative_data[executor.BEST_CUMULATIVE_SCORE],
                     expected_output)
    self.assertEqual(best_tuner_ix, 0)
    self.assertEqual(cumulative_data[executor.OBJECTIVE_DIRECTION], 'min')

    data1[executor.BEST_CUMULATIVE_SCORE] = [3.5, 2, 1.0]
    data2[executor.BEST_CUMULATIVE_SCORE] = [4.0, 0.5, 0.3, 0.2]
    (cumulative_data, best_tuner_ix) = executor.merge_trial_data(data1, data2)
    expected_output = [3.5, 2, 1.0, 1.0, 0.5, 0.3, 0.2]

    self.assertEqual(cumulative_data[executor.BEST_CUMULATIVE_SCORE],
                     expected_output)
    self.assertEqual(best_tuner_ix, 1)
    self.assertEqual(cumulative_data[executor.OBJECTIVE_DIRECTION], 'min')


if __name__ == '__main__':
  absltest.main()
