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
"""Tests for nitroml.components.meta_learning.meta_feature_gen.component."""

from typing import Text

from absl.testing import absltest
from nitroml.components import MetaFeatureGen
from nitroml.components.meta_learning import artifacts
from tfx.orchestration import data_types
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

  def _verify_outputs(self, meta_feature_gen):
    self.assertEqual(artifacts.MetaFeatures.TYPE_NAME,
                     meta_feature_gen.outputs['meta_features'].type_name)

  def testConstruct(self):
    meta_feature_gen = MetaFeatureGen(
        statistics=self.statistics,
        custom_config=self.custom_config)
    self._verify_outputs(meta_feature_gen)

  def testConstructWithExamples(self):
    meta_feature_gen = MetaFeatureGen(
        statistics=self.statistics,
        transformed_examples=self.examples,
        custom_config=self.custom_config)
    self._verify_outputs(meta_feature_gen)


if __name__ == '__main__':
  absltest.main()
