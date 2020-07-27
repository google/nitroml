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

from nitroml.components import metafeature_gen
from nitroml.components import metalearner
from nitroml import autodata
from tfx.components.base import base_component
from tfx import types
import kerastuner

from nitroml.components.transform import component as transform


class MetaLearningWrapper(object):
  """A helper class that wraps definition of meta learning sub-pipeline."""

  def __init__(self,
               train_autodata_list: List[autodata.AutoData],
               meta_train_data: Dict[str, Any],
               algorithm: str = 'majority_voting'):

    self._train_autodata_list = train_autodata_list
    self._meta_train_data = meta_train_data
    self._pipeline = []
    self._algorithm = algorithm
    self._recommended_search_space = None
    self._build_metalearner()

  @property
  def pipeline(self) -> List[base_component.BaseComponent]:
    return self._pipeline

  @property
  def recommended_search_space(self) -> kerastuner.HyperParameters:
    return self._recommended_search_space

  def _build_metalearner(self) -> None:
    """Builds the meta-learning pipeline."""

    self._pipeline = []

    for ix, autodata in enumerate(self._train_autodata_list):
      metafeature_gen = self._create_metafeature_gen(
          statistics=autodata.statistics,
          transformed_examples=autodata.transformed_examples,
          instance_name=f'train_{autodata.instance_name}')
      self._pipeline.append(metafeature_gen)
      self._meta_train_data[
          f'meta_train_features_{ix}'] = metafeature_gen.outputs.metafeatures

    learner = metalearner.MetaLearner(
        algorithm=self._algorithm, **self._meta_train_data)
    self._pipeline.append(learner)
    self._recommended_search_space = learner.outputs.meta_hyperparameters

  def _create_metafeature_gen(self,
                              statistics: types.Channel,
                              transformed_examples: Optional[
                                  types.Channel] = None,
                              instance_name: str = None) -> types.Channel:
    """Creates and returns the `MetaFeatureGen` component.

      Args:
        statistics: Channel containing the dataset statistics proto path.
        transformed_examples: Channel containing the transformed examples paths.

      Returns:
        The MetaFeatureGen component
    """

    return metafeature_gen.MetaFeatureGen(
        statistics=statistics,
        transformed_examples=(transformed_examples),
        instance_name=instance_name)