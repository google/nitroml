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
"""Tests for nitroml.automl.metalearning.metalearner.component."""

from absl.testing import absltest
from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.metalearner.component import MetaLearner
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(absltest.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()

    num_train = 5
    self.meta_train_data = {}
    for ix in range(num_train):
      self.meta_train_data[f'hparams_train_{ix}'] = channel_utils.as_channel(
          [standard_artifacts.HyperParameters()])
      self.meta_train_data[
          f'meta_train_features_{ix}'] = channel_utils.as_channel(
              [artifacts.MetaFeatures()])

    self.custom_config = {'some': 'thing', 'some other': 1, 'thing': 2}

  def testConstructWithMajorityVoting(self):

    metalearner = MetaLearner(
        algorithm='majority_voting',
        custom_config=self.custom_config,
        **self.meta_train_data)
    self.assertEqual(artifacts.KCandidateHyperParameters.TYPE_NAME,
                     metalearner.outputs['output_hyperparameters'].type_name)
    self.assertEqual(standard_artifacts.Model.TYPE_NAME,
                     metalearner.outputs['metamodel'].type_name)

  def testConstructWithNearestNeighbor(self):

    metalearner = MetaLearner(
        algorithm='nearest_neighbor',
        custom_config=self.custom_config,
        **self.meta_train_data)
    self.assertEqual(artifacts.KCandidateHyperParameters.TYPE_NAME,
                     metalearner.outputs['output_hyperparameters'].type_name)
    self.assertEqual(standard_artifacts.Model.TYPE_NAME,
                     metalearner.outputs['metamodel'].type_name)


if __name__ == '__main__':
  absltest.main()
