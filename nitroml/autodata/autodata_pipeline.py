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
"""The AutoData pipeline for automatically preprocessing data for trainers."""

from typing import List, Optional

from absl import logging
from nitroml.autodata.preprocessors import basic_preprocessor
from nitroml.autodata.preprocessors import preprocessor as pp
from nitroml.components.transform import component as transform
from tfx import components as tfx
from tfx import types
from tfx.components.base import base_component

from nitroml.protos import problem_statement_pb2 as ps_pb2


class AutoData:
  """AutoData preprocesses raw data into artifacts for trainers.

  It is intended to be prepended to a pipeline where components like trainers
  and tuners need access to a dataset's schema, feature statistics, and
  preprocessed features.

  Trainers and tuners should consume the outputs of the AutoData pipeline
  through trainer adapters defined in the `autodata.trainer_adapters` package.
  """

  # TODO(b/162070520): Migrate this to the TFX sub-pipeline API when available.

  def __init__(self,
               problem_statement: ps_pb2.ProblemStatement,
               examples: types.Channel,
               preprocessor: Optional[pp.Preprocessor] = None,
               instance_name: Optional[str] = None):
    """Constructs an AutoDataPipeline instance.

    Args:
      problem_statement: The problem statement proto that defines the task. This
        is used to identify and preprocess the label (required).
      examples: A Channel of type `standard_artifacts.Examples` (required). This
        should contain the two splits 'train' and 'eval'.
      preprocessor: A `Preprocessor` instance which defines how TensorFlow
        Transform should preprocess raw Tensors from tensorflow.Examples.
      instance_name: Optional unique instance name. Necessary iff multiple
        AutoDataPipeline instances are declared in the same pipeline.
    """


    if not preprocessor:
      logging.info('Using default preprocessor: BasicPreprocessor.')
      preprocessor = basic_preprocessor.BasicPreprocessor()

    self._preprocessor = preprocessor

    autodata_instance_name = 'AutoData'
    if instance_name:
      autodata_instance_name = f'{autodata_instance_name}.{instance_name}'

    # Computes statistics over data for visualization and example validation.
    self._statistics_gen = self._build_statistics_gen(examples,
                                                      autodata_instance_name)

    # Generates schema based on statistics files.
    self._schema_gen = self._build_schema_gen(
        self._statistics_gen.outputs.statistics, autodata_instance_name)

    self._transform = self._build_transform(problem_statement, examples,
                                            self._schema_gen.outputs.schema,
                                            autodata_instance_name)

  @property
  def components(self) -> List[base_component.BaseComponent]:
    """Return the AutoData pipeline's constituent components."""

    return [self._schema_gen, self._statistics_gen, self._transform]

  @property
  def statistics(self) -> types.Channel:
    """Channel containing the dataset statistics proto path."""

    return self._statistics_gen.outputs.statistics

  @property
  def schema(self) -> types.Channel:
    """Channel containing the dataset schema proto path."""

    return self._schema_gen.outputs.schema

  @property
  def transformed_examples(self) -> types.Channel:
    """Channel containing the transformed examples paths."""

    return self._transform.outputs.transformed_examples

  @property
  def transform_graph(self) -> types.Channel:
    """Channel containing the transform output path."""

    return self._transform.outputs.transform_graph

  def _build_statistics_gen(self, examples: types.Channel,
                            instance_name: Optional[str]) -> tfx.StatisticsGen:
    """Returns the StatisticsGen component."""

    # TODO(b/156134844): Allow passing TFDV StatsOptions to automatically infer
    # useful semantic types
    return tfx.StatisticsGen(examples=examples, instance_name=instance_name)

  def _build_schema_gen(self, statistics: types.Channel,
                        instance_name: Optional[str]) -> tfx.SchemaGen:
    """Returns the SchemaGen component."""

    return tfx.SchemaGen(
        statistics=statistics,
        infer_feature_shape=self._preprocessor.requires_inferred_feature_shapes,
        instance_name=instance_name)

  def _build_transform(self, problem_statement: ps_pb2.ProblemStatement,
                       examples: types.Channel, schema: types.Channel,
                       instance_name: Optional[str]) -> transform.Transform:
    """Returns the Transform component."""

    # TODO(b/148932926) We use NitroML's Transform component, so that we can
    # forward the problem statement and schema to the preprocessing_fn. Use
    # TFX:OSS instead.
    return transform.Transform(
        examples=examples,
        schema=schema,
        preprocessing_fn=self._preprocessor.preprocessing_fn,
        custom_config=self._preprocessor.custom_config(
            problem_statement=problem_statement),
        instance_name=instance_name)
