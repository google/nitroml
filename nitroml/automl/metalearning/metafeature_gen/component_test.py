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
"""Tests for nitroml.automl.metalearning.metafeature_gen.component."""

from absl.testing import absltest
from nitroml.automl.metalearning import artifacts
from nitroml.automl.metalearning.metafeature_gen.component import MetaFeatureGen
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(absltest.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()

    examples_artifact = standard_artifacts.Examples()
    examples_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    statistics_artifact = standard_artifacts.ExampleStatistics()
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        ['train'])

    self.examples = channel_utils.as_channel([examples_artifact])
    self.statistics = channel_utils.as_channel([statistics_artifact])
    self.custom_config = {'some': 'thing', 'some other': 1, 'thing': 2}

  def _verify_outputs(self, metafeature_gen):
    self.assertEqual(artifacts.MetaFeatures.TYPE_NAME,
                     metafeature_gen.outputs['metafeatures'].type_name)

  def testConstruct(self):
    metafeature_gen = MetaFeatureGen(
        statistics=self.statistics, custom_config=self.custom_config)
    self._verify_outputs(metafeature_gen)

  def testConstructWithExamples(self):
    metafeature_gen = MetaFeatureGen(
        statistics=self.statistics,
        transformed_examples=self.examples,
        custom_config=self.custom_config)
    self._verify_outputs(metafeature_gen)


if __name__ == '__main__':
  absltest.main()
