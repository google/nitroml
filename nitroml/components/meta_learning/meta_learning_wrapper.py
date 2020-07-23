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
r"""Meta Learning helper class the defines the meta learning DAG."""

from typing import Any, Dict, Optional, Text, List

from nitroml.components import MetaFeatureGen
from nitroml.components import MetaLearner
from tfx.components.base import base_component
from tfx import types


class MetaLearningWrapper(object):
  """A helper class that wraps definition of meta learning sub-pipeline."""

  def __init__(self,
               train_transformed_examples: List[base_component.BaseComponent],
               train_stats_gens: List[base_component.BaseComponent],
               test_transformed_examples: List[base_component.BaseComponent],
               test_stats_gens: List[base_component.BaseComponent],
               algorithm: Text = 'nearest_neighbor'):

    self._train_transformed_examples = train_transformed_examples
    self._train_stats_gens = train_stats_gens
    self._test_transformed_examples = test_transformed_examples
    self._test_stats_gens = test_stats_gens
    self._algorithm = algorithm
    self._pipeline = []
    self._build_pipeline()

  @property
  def pipeline(self) -> List[base_component.BaseComponent]:
    return self._pipeline

  # TODO(nikhilmehta): Add instance_name.
  def _build_pipeline(self) -> None:
    """Builds the meta-learning pipeline."""

    self._pipeline = []
    train_meta_features = {}
    for ix, stats_gen in enumerate(self._train_stats_gens):
      meta_feature_gen = MetaFeatureGen(
          statistics=stats_gen.outputs.statistics, instance_name='1')
      train_meta_features[
          f'meta_train_features_{ix}'] = meta_feature_gen.outputs.meta_features
      self._pipeline.append(meta_feature_gen)

    learner = MetaLearner(algorithm='nearest_neighbor', **train_meta_features)
    self._pipeline.append(learner)

    test_statistics = {}
    for ix, stats_gen in enumerate(self._test_stats_gens):
      meta_feature_gen = MetaFeatureGen(
          statistics=stats_gen.outputs.statistics, instance_name='2')
      self._pipeline.append(meta_feature_gen)
