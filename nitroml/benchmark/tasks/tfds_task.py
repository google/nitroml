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
r"""A dataset from a TFDS Dataset."""

from typing import List

from absl import logging
import nitroml
import tensorflow_datasets as tfds
from tfx import components as tfx
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.proto import example_gen_pb2
from tfx.utils.dsl_utils import external_input

from nitroml.protos import problem_statement_pb2 as ps_pb2


class TFDSTask(nitroml.BenchmarkTask):
  """A NitroML BenchmarkTask from a TensorFlow Datasets (TFDS) Dataset."""

  def __init__(self, dataset_builder: tfds.core.DatasetBuilder):
    """A NitroML dataset from a TFDS DatasetBuilder.

    Args:
      dataset_builder: A `tfds.DatasetBuilder` instance which defines the
        TFDS dataset to use. Example: `dataset =
          TFDSTask(tfds.builder('titanic'))`
    """

    # TODO(b/159086401): Download and prepare the dataset in a component
    # instead of at construction time, so that this step happens lazily during
    # pipeline execution.
    logging.info('Preparing dataset...')
    dataset_builder.download_and_prepare()
    logging.info(dataset_builder.info)

    self._dataset_builder = dataset_builder
    self._example_gen = self._make_example_gen()

  def _make_example_gen(self) -> base_component.BaseComponent:
    """Returns a TFX ExampleGen which produces the desired split."""

    splits = []
    for name, value in self._dataset_builder.info.splits.items():
      # Assume there is only one file per split.
      # Filename will be like `'fashion_mnist-test.tfrecord-00000-of-00001'`.
      assert len(value.filenames) == 1
      pattern = value.filenames[0]
      splits.append(example_gen_pb2.Input.Split(name=name, pattern=pattern))

    logging.info('Splits: %s', splits)
    input_config = example_gen_pb2.Input(splits=splits)
    return tfx.ImportExampleGen(
        input=external_input(self._dataset_builder.data_dir),
        input_config=input_config)

  @property
  def name(self) -> str:
    return self._dataset_builder.info.name

  @property
  def components(self) -> List[base_component.BaseComponent]:
    return [self._example_gen]

  @property
  def train_and_eval_examples(self) -> types.Channel:
    """Returns train and eval labeled examples."""

    return self._example_gen.outputs.examples

  @property
  def test_examples(self) -> types.Channel:
    """Returns test labeled examples."""

    return self._example_gen.outputs.examples

  @property
  def test_example_splits(self) -> List[str]:
    return ['eval']

  @property
  def problem_statement(self) -> ps_pb2.ProblemStatement:
    """Returns the ProblemStatement associated with this Task."""

    # Supervised keys is a two-tuple.
    _, target_key = self._dataset_builder.info.supervised_keys
    return ps_pb2.ProblemStatement(
        owner=['nitroml'],
        tasks=[
            ps_pb2.Task(
                name=self.name,
                type=ps_pb2.Type(
                    binary_classification=ps_pb2.BinaryClassification(
                        label=target_key)),
            )
        ])
