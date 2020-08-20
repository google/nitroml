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

from typing import Any, Dict, Optional, List, Tuple

from nitroml.autodata.autodata_pipeline import AutoData
from nitroml.components.metalearning.metafeature_gen.component import MetaFeatureGen
from nitroml.components.metalearning.metalearner import component as metalearner
from nitroml.components.metalearning.tuner import component as tuner_component
from tfx import types
from tfx.components.base import base_component
from tfx.proto import trainer_pb2

from google.protobuf import text_format


class MetaLearningWrapper(object):
  """A helper class that wraps definition of meta learning sub-pipeline."""

  def __init__(self,
               train_autodata_list: List[AutoData],
               meta_train_data: Dict[str, Any],
               algorithm: str = 'majority_voting'):

    self._train_autodata_list = train_autodata_list
    self._meta_train_data = meta_train_data
    self._pipeline = []
    self._algorithm = algorithm
    self._recommended_search_space = None
    self._metamodel = None
    self._build_metalearner()

  @property
  def pipeline(self) -> List[base_component.BaseComponent]:
    return self._pipeline

  @property
  def recommended_search_space(self) -> types.Channel:
    return self._recommended_search_space

  @property
  def metamodel(self) -> types.Channel:
    return self._metamodel

  def _build_metalearner(self) -> None:
    """Builds the meta-learning pipeline."""

    self._pipeline = []

    # Majority-Voting algorithm produces task-independent recommendations.
    # We do not need meta-features for this algorithm.
    if self._algorithm not in ['majority_voting']:
      # Add metafeature_gen components
      for ix, autodata in enumerate(self._train_autodata_list):
        instance_name = autodata.id.replace('AutoData.', '')
        metafeature_gen = self._create_metafeature_gen(
            statistics=autodata.statistics,
            transformed_examples=autodata.transformed_examples,
            instance_name=instance_name)
        self._pipeline.append(metafeature_gen)
        self._meta_train_data[
            f'meta_train_features_{ix}'] = metafeature_gen.outputs.metafeatures

    learner = metalearner.MetaLearner(
        algorithm=self._algorithm, **self._meta_train_data)
    self._pipeline.append(learner)
    self._recommended_search_space = learner.outputs.output_hyperparameters
    self._metamodel = learner.outputs.metamodel

  def create_test_components(
      self, test_autodata: AutoData, tuner_steps: int
  ) -> Tuple[List[base_component.BaseComponent], types.Channel]:
    """Returns the list of components required for test dataset metalearning subpipipeline.

    Args:
      test_autodata: Autodata of the test subpipeline.
      tuner_steps: Number of steps to tune.

    Returns:
      The list of components that compose the subpipeline and the best hparams
      channel.
    """

    instance_name = test_autodata.id.replace('AutoData.', '')
    meta_test_pipeline = []
    test_metafeature = None
    if self._algorithm not in ['majority_voting']:
      metafeature_gen = self._create_metafeature_gen(
          statistics=test_autodata.statistics,
          transformed_examples=test_autodata.transformed_examples,
          instance_name=instance_name)
      test_metafeature = metafeature_gen.outputs.metafeatures
      meta_test_pipeline.append(metafeature_gen)

    tuner = tuner_component.AugmentedTuner(
        tuner_fn='examples.auto_trainer.tuner_fn',
        examples=test_autodata.transformed_examples,
        transform_graph=test_autodata.transform_graph,
        train_args=trainer_pb2.TrainArgs(num_steps=tuner_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=1),
        metalearning_algorithm=self._algorithm,
        warmup_hyperparameters=self.recommended_search_space,
        metamodel=self.metamodel,
        metafeature=test_metafeature,
        custom_config={
            # Pass the problem statement proto as a text proto. Required
            # since custom_config must be JSON-serializable.
            'problem_statement':
                text_format.MessageToString(
                    message=test_autodata.problem_statement, as_utf8=True),
        },
        instance_name=instance_name)
    meta_test_pipeline += [tuner]

    return meta_test_pipeline, tuner.outputs.best_hyperparameters

  def _create_metafeature_gen(self,
                              statistics: types.Channel,
                              transformed_examples: Optional[
                                  types.Channel] = None,
                              instance_name: str = None) -> MetaFeatureGen:
    """Creates and returns the `MetaFeatureGen` component.

    Args:
      statistics: Channel containing the dataset statistics proto path.
      transformed_examples: Channel containing the transformed examples paths.
      instance_name: Optional unique instance name. Necessary iff multiple
        MetaFeatureGen components are declared in the same pipeline.

    Returns:
      The MetaFeatureGen component
    """

    return MetaFeatureGen(
        statistics=statistics,
        transformed_examples=(transformed_examples),
        instance_name=instance_name)
