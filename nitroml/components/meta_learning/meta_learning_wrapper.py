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
import kerastuner


class MetaLearningWrapper(object):
  """A helper class that wraps definition of meta learning sub-pipeline."""

  def __init__(self,
               train_transformed_examples: List[base_component.BaseComponent],
               train_stats_gens: List[base_component.BaseComponent],
               meta_train_data: Dict[str, Any],
               algorithm: str = 'majority_voting'):

    self._train_transformed_examples = train_transformed_examples
    self._train_stats_gens = train_stats_gens
    self._meta_train_data = meta_train_data
    self._pipeline = []
    self._algorithm = algorithm
    self._recommended_search_space = None
    self._build_meta_learner()

  @property
  def pipeline(self) -> List[base_component.BaseComponent]:
    return self._pipeline

  @property
  def recommended_search_space(self) -> kerastuner.HyperParameters:
    return self._recommended_search_space

  def _build_meta_learner(self) -> None:
    """Builds the meta-learning pipeline."""

    self._pipeline = []
    train_meta_features = {}
    for ix, stats_gen in enumerate(self._train_stats_gens):
      self._meta_train_data[
          f'meta_train_features_{ix}'] = self._get_meta_feature_channel(
              stats_gen, instance_name=f'train_{ix}')

    learner = MetaLearner(algorithm=self._algorithm, **self._meta_train_data)
    self._pipeline.append(learner)
    self._recommended_search_space = learner.outputs.meta_hyperparameters

  def _get_meta_feature_channel(self,
                                statistics_gen: base_component.BaseComponent,
                                transform: Optional[
                                    base_component.BaseComponent] = None,
                                instance_name: str = None) -> types.Channel:
    """Creates the `MetaFeatureGen` component and returns the output channel.

      Args:
        statistics_gen: The tfx StatisticsGen component to create MetaFeatures.
        transformed_examples: The tfx Transform component

      Returns:
        meta_features: MetaFeatures channel
    """

    meta_feature_gen = MetaFeatureGen(
        statistics=statistics_gen.outputs.statistics,
        transformed_examples=(transform.outputs.transformed_examples
                              if transform else None),
        instance_name=instance_name)
    self._pipeline.append(meta_feature_gen)
    return meta_feature_gen.outputs.meta_features