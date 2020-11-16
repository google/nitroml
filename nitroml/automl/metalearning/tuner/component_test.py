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
"""Tests for nitroml.components.tuner.component."""

from absl.testing import absltest
from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.tuner import component as tuner_component
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(absltest.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()

    examples_artifact = standard_artifacts.Examples()
    examples_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    transform_output = standard_artifacts.TransformGraph()

    self.examples = channel_utils.as_channel([examples_artifact])
    self.schema = channel_utils.as_channel([standard_artifacts.Schema()])
    self.transform_graph = channel_utils.as_channel([transform_output])
    self.custom_config = {'some': 'thing', 'some other': 1, 'thing': 2}
    self.train_args = trainer_pb2.TrainArgs(num_steps=100)
    self.eval_args = trainer_pb2.EvalArgs(num_steps=50)
    self.tune_args = tuner_pb2.TuneArgs(num_parallel_trials=3)
    self.warmup_hyperparams = channel_utils.as_channel(
        [artifacts.KCandidateHyperParameters()])
    self.meta_model = channel_utils.as_channel([standard_artifacts.Model()])

  def _verify_outputs(self, tuner):
    self.assertEqual(standard_artifacts.HyperParameters.TYPE_NAME,
                     tuner.outputs['best_hyperparameters'].type_name)
    self.assertEqual(tuner_component.TunerData.TYPE_NAME,
                     tuner.outputs['trial_summary_plot'].type_name)

  def testConstructWithModuleFile(self):
    tuner = tuner_component.AugmentedTuner(
        examples=self.examples,
        schema=self.schema,
        transform_graph=self.transform_graph,
        train_args=self.train_args,
        eval_args=self.eval_args,
        tune_args=self.tune_args,
        module_file='some/random/path',
        custom_config=self.custom_config)
    self._verify_outputs(tuner)

  def testConstructWithTunerFn(self):
    tuner = tuner_component.AugmentedTuner(
        examples=self.examples,
        schema=self.schema,
        transform_graph=self.transform_graph,
        train_args=self.train_args,
        eval_args=self.eval_args,
        tune_args=self.tune_args,
        module_file='some/random/path/tuner_fn',
        custom_config=self.custom_config)
    self._verify_outputs(tuner)

  def testConstructWithWarmStarting(self):
    tuner = tuner_component.AugmentedTuner(
        examples=self.examples,
        schema=self.schema,
        transform_graph=self.transform_graph,
        train_args=self.train_args,
        eval_args=self.eval_args,
        tune_args=self.tune_args,
        module_file='some/random/path/tuner_fn',
        metalearning_algorithm='max_voting',
        warmup_hyperparameters=self.warmup_hyperparams,
        custom_config=self.custom_config)
    self._verify_outputs(tuner)

  def testConstructWithNearestNeighbor(self):
    tuner = tuner_component.AugmentedTuner(
        examples=self.examples,
        schema=self.schema,
        transform_graph=self.transform_graph,
        train_args=self.train_args,
        eval_args=self.eval_args,
        tune_args=self.tune_args,
        module_file='some/random/path/tuner_fn',
        metalearning_algorithm='nearest_neighbor',
        warmup_hyperparameters=self.warmup_hyperparams,
        metamodel=self.meta_model,
        custom_config=self.custom_config)
    self._verify_outputs(tuner)


if __name__ == '__main__':
  absltest.main()
