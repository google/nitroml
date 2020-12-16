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
"""A Keras Model trainer adapter for AutoData.

The consumed artifacts include:
 * TensorFlow Transform outputs.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import kerastuner
from nitroml.automl.autodata.trainer_adapters import estimator_adapter as ea
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft

from nitroml.protos import problem_statement_pb2 as ps_pb2

FeatureColumn = Any


class KerasModelAdapter:
  """Creates feature columns and specs from TFX artifacts."""

  SPARSE_CATEGORICAL_CE = 'sparse_categorical_crossentropy'
  BINARY_CE = 'binary_crossentropy'

  def __init__(self, problem_statement: ps_pb2.ProblemStatement,
               transform_graph_dir: str):
    """Initializes the DataProvider from TFX artifacts.

    Args:
      problem_statement: Defines the task and label key.
      transform_graph_dir: Path to the TensorFlow Transform graph artifacts.
    """

    self._problem_statement = problem_statement
    # Parse transform.
    self._tf_transform_output = tft.TFTransformOutput(transform_graph_dir)
    # Parse schema.
    self._dataset_schema = self._tf_transform_output.transformed_metadata.schema

    # We compose the KerasAdapeter with an EstimatorAdapter and convert its
    # methods into Keras-specific ones.
    self._estimator_adapter = ea.EstimatorAdapter(problem_statement,
                                                  transform_graph_dir)

  @property
  def raw_label_key(self) -> str:
    """The raw label key as defined in the ProblemStatement."""

    return self._estimator_adapter.raw_label_key

  @property
  def transformed_label_key(self) -> str:
    """The label key after applying TensorFlow Transform to the Examples."""

    return self._estimator_adapter.transformed_label_key

  @property
  def head_size(self) -> int:
    """Returns the head size for this task."""

    return self._estimator_adapter.head.logits_dimension

  @property
  def head_activation(self) -> Optional[str]:
    """Returns the activation for the final layer."""

    # TODO(nikhilmehta): Change the task object to allow label_key to be a list.
    task_type = self._problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return 'softmax'
    if task_type.HasField('binary_classification'):
      return 'sigmoid'
    if task_type.HasField('one_dimensional_regression'):
      return None
    raise ValueError('Invalid task type: {}'.format(task_type))

  @property
  def loss(self) -> str:
    """Returns the keras loss_fn for this task."""

    task_type = self._problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return 'sparse_categorical_crossentropy'
    if task_type.HasField('binary_classification'):
      return 'binary_crossentropy'
    if task_type.HasField('one_dimensional_regression'):
      return 'mse'
    raise ValueError('Invalid task type: {}'.format(task_type))

  @property
  def metrics(self) -> List[tf.keras.metrics.Metric]:
    """Returns the keras metrics for evaluation."""

    task_type = self._problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return [
          tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
          tf.keras.metrics.SparseCategoricalCrossentropy(name='average_loss')
      ]
    if task_type.HasField('binary_classification'):
      return [
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.BinaryCrossentropy(name='average_loss'),
          tf.keras.metrics.AUC()
      ]
    if task_type.HasField('one_dimensional_regression'):
      return [
          tf.keras.metrics.MeanSquaredError(name='mse'),
      ]
    raise ValueError('Invalid task type: {}'.format(task_type))

  @property
  def tuner_objective(self) -> kerastuner.Objective:
    """Returns the target objective of the tuner."""

    # TODO(github.com/google/nitroml/issues/29): Support Regression tasks.
    if self.head_size == 1:
      return kerastuner.Objective('val_auc', 'max')
    else:
      return kerastuner.Objective('val_accuracy', 'max')

  def get_input_layers(self) -> Dict[str, tf.keras.layers.Input]:
    """Returns input layers for a Keras Model."""

    feature_spec = self._tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(self.transformed_label_key)
    input_layers = {}
    for colname, spec in feature_spec.items():
      input_layers[colname] = tf.keras.layers.Input(
          name=colname, shape=spec.shape, dtype=spec.dtype)
    return input_layers

  def get_serving_input_receiver_fn(
      self) -> Callable[[], tf.estimator.export.ServingInputReceiver]:
    """Returns the serving_input_receiver_fn used when exporting a SavedModel.

    Returns:
      An serving_input_receiver_fn. Its returned ServingInputReceiver takes as
      input a batch of raw (i.e. untransformed) serialized tensorflow.Examples.
      The model can be used for serving or for batch inference.
    """

    return self._estimator_adapter.get_serving_input_receiver_fn()

  def get_eval_input_receiver_fn(
      self) -> Callable[[], tfma.export.EvalInputReceiver]:
    """Returns the eval_input_receiver_fn to build a SavedModel for TFMA.

    Returns:
      An eval_input_receiver_fn. Its returned EvalInputReceiver takes as
      input a batch of raw (i.e. untransformed) serialized tensorflow.Examples.
      The model will compute predictions and metrics that TFMA can consume.
    """
    return self._estimator_adapter.get_eval_input_receiver_fn()

  def get_dense_feature_columns(self) -> List[FeatureColumn]:
    """Returns the lists of dense FeatureColumns to be fed into the model."""

    return self._estimator_adapter.get_dense_feature_columns()

  def get_sparse_feature_columns(self) -> List[FeatureColumn]:
    """Returns the list of sparse FeatureColumns to be fed into the model."""

    return self._estimator_adapter.get_sparse_feature_columns()

  def get_dataset(self,
                  file_pattern: Union[str, List[str]],
                  batch_size: int,
                  num_epochs: Optional[int] = None,
                  shuffle: Optional[bool] = True,
                  shuffle_buffer_size: int = 10000,
                  shuffle_seed: Optional[int] = None,
                  prefetch_buffer_size: Optional[int] = None,
                  reader_num_threads: Optional[int] = None,
                  parser_num_threads: Optional[int] = None,
                  sloppy_ordering: bool = False,
                  drop_final_batch: bool = False) -> tf.data.Dataset:
    """Returns an input_fn that returns a `tf.data.Dataset` from Examples.

    Args:
      file_pattern: List of files or patterns of file paths containing Example
        records. See tf.io.gfile.glob for pattern rules.
      batch_size: An int representing the number of records to combine in a
        single batch.
      num_epochs: Integer specifying the number of times to read through the
        dataset. If None, cycles through the dataset forever. Defaults to None.
      shuffle: A boolean, indicates whether the input should be shuffled.
        Defaults to True.
      shuffle_buffer_size: Buffer size of the ShuffleDataset. A large capacity
        ensures better shuffling but would increase memory usage and startup
        time.
      shuffle_seed: Randomization seed to use for shuffling.
      prefetch_buffer_size: Number of feature batches to prefetch in order to
        improve performance. Recommended value is the number of batches consumed
        per training step. Defaults to auto-tune.
      reader_num_threads: Number of threads used to read Example records. If >1,
        the results will be interleaved. Defaults to 1.
      parser_num_threads: Number of threads to use for parsing Example tensors
        into a dictionary of Feature tensors. Defaults to 2.
      sloppy_ordering: If True, reading performance will be improved at the cost
        of non-deterministic ordering. If False, the order of elements produced
        is deterministic prior to shuffling (elements are still randomized if
        shuffle=True. Note that if the seed is set, then order of elements after
        shuffling is deterministic). Defaults to False.
      drop_final_batch: If True, and the batch size does not evenly divide the
        input dataset size, the final smaller batch will be dropped. Defaults to
        False.

    Returns:
      Returns a `tf.data.Dataset` of (dict of feature string to Tensor, label
      Tensor).
    """

    input_fn = self._estimator_adapter.get_input_fn(
        file_pattern=file_pattern,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=shuffle_seed,
        prefetch_buffer_size=prefetch_buffer_size,
        reader_num_threads=reader_num_threads,
        parser_num_threads=parser_num_threads,
        sloppy_ordering=sloppy_ordering,
        drop_final_batch=drop_final_batch)
    return input_fn()
