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
"""An Estimator trainer that trains on any tabular dataset.

The consumed artifacts include:
 * Dataset schema.
 * TensorFlow Transform outputs.
"""

from typing import Any, Callable, List, Optional, Union

import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft

from nitroml.protos import problem_statement_pb2 as ps_pb2
from tensorflow_metadata.proto.v0 import schema_pb2

FeatureColumn = Any


class EstimatorAdapter:
  """Creates feature columns and specs from TFX artifacts."""

  def __init__(self, problem_statement: ps_pb2.ProblemStatement,
               transform_graph_dir: str):
    """Initializes the EstimatorAdapter from TFX artifacts.

    Args:
      problem_statement: Defines the task and label key.
      transform_graph_dir: Path to the TensorFlow Transform graph artifacts.
    """

    self._problem_statement = problem_statement
    # Parse transform.
    self._tf_transform_output = tft.TFTransformOutput(transform_graph_dir)
    # Parse schema.
    self._dataset_schema = self._tf_transform_output.transformed_metadata.schema

  @property
  def raw_label_key(self) -> str:
    """The raw label key as defined in the ProblemStatement."""

    # TODO(nikhilmehta): Change the task object to allow label_key to be a list.
    task_type = self._problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return task_type.multi_class_classification.label
    if task_type.HasField('binary_classification'):
      return task_type.binary_classification.label
    if task_type.HasField('one_dimensional_regression'):
      return task_type.one_dimensional_regression.label
    raise ValueError('Invalid task type: {}'.format(task_type))

  @property
  def transformed_label_key(self) -> str:
    """The label key after applying TensorFlow Transform to the Examples."""

    return self.raw_label_key

  @property
  def head(self) -> tf.estimator.Head:
    """Returns the Estimator Head for this task."""

    task_type = self._problem_statement.tasks[0].type
    if task_type.HasField('one_dimensional_regression'):
      return tf.estimator.RegressionHead()
    num_classes = (
        self._tf_transform_output.num_buckets_for_transformed_feature(
            self.raw_label_key))
    if task_type.HasField('multi_class_classification'):
      return tf.estimator.MultiClassHead(num_classes)
    if task_type.HasField('binary_classification'):
      return tf.estimator.BinaryClassHead()
    raise ValueError('Invalid task type: {}'.format(task_type))

  def _get_numeric_feature_columns(self,
                                   include_integer_columns: bool = False
                                  ) -> List[FeatureColumn]:
    """Creates a set of feature columns.

    Args:
      include_integer_columns: Whether integer columns in the examples should be
        included in the numeric columns as floats.

    Returns:
      A list of feature columns.
    """

    numeric_columns = []
    for feature in self._dataset_schema.feature:

      feature_name = feature.name
      if feature_name == self.raw_label_key:
        continue

      feature_storage_type = _get_feature_storage_type(self._dataset_schema,
                                                       feature_name)

      if feature_storage_type == tf.int64 and not include_integer_columns:
        continue

      # NOTE: Int features are treated as both numerical and categorical. For
      # example MNIST stores its features as int16 features, but are continuous.
      if feature_storage_type == tf.float32 or feature_storage_type == tf.int64:

        # Numerical feature.
        dim = _get_feature_dim(self._dataset_schema, feature_name)

        # Numerical feature normalized in 0-1.
        current_feature = tf.feature_column.numeric_column(
            feature_name, shape=(dim,), dtype=feature_storage_type)
        numeric_columns.append(current_feature)
    return numeric_columns

  def _get_sparse_categorical_feature_columns(
      self, include_integer_columns: bool = True) -> List[FeatureColumn]:
    """Creates a set of sparse categorical feature columns.

    Args:
      include_integer_columns: Whether integer columns in the examples should be
        included in the categorical columns.

    Returns:
      A list of feature columns.
    """

    feature_columns = []
    for feature in self._dataset_schema.feature:

      feature_name = feature.name
      if feature_name == self.raw_label_key:
        continue

      feature_storage_type = _get_feature_storage_type(self._dataset_schema,
                                                       feature_name)

      if feature_storage_type == tf.float32:
        continue

      if feature_storage_type == tf.int64:
        if not include_integer_columns:
          continue

        # Categorical or categorical-set feature stored as an integer(s).
        num_buckets = (
            self._tf_transform_output.num_buckets_for_transformed_feature(
                feature_name))
        new_feature_column = tf.feature_column.categorical_column_with_identity(
            feature_name, num_buckets=num_buckets)
      elif feature_storage_type == tf.string:
        # Note TFT automatically converts string columns to int columns.
        raise ValueError(
            'String dtypes should be converted to int columns by Transform')
      else:
        raise ValueError(f'Unsupported dtype: {feature_storage_type}')
      feature_columns.append(new_feature_column)
    return feature_columns

  def _get_embedding_feature_columns(self,
                                     include_integer_columns: bool = True
                                    ) -> List[FeatureColumn]:
    """Creates a set of embedding feature columns.

    Args:
      include_integer_columns: Whether integer columns in the examples should be
        included in the embeddings.

    Returns:
      A list of feature columns.
    """

    return [
        tf.feature_column.embedding_column(column, dimension=10) for column in
        self._get_sparse_categorical_feature_columns(include_integer_columns)
    ]

  def get_dense_feature_columns(self) -> List[FeatureColumn]:
    """Returns the lists of dense FeatureColumns to be fed into the model."""

    return self._get_numeric_feature_columns(
    ) + self._get_embedding_feature_columns()

  def get_sparse_feature_columns(self) -> List[FeatureColumn]:
    """Returns the list of sparse FeatureColumns to be fed into the model."""

    return self._get_numeric_feature_columns(
    ) + self._get_sparse_categorical_feature_columns()

  def get_input_fn(
      self,
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
      drop_final_batch: bool = False,
      max_num_batches: Optional[int] = None) -> Callable[[], tf.data.Dataset]:
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
      max_num_batches: The maximum number of batches to include in the dataset.
        Generally used to subsample a large dataset when `num_epoch=1`.

    Returns:
      Returns an input_fn that returns a `tf.data.Dataset`.
    """

    # Since we're not applying the transform graph here, we're using Transform
    # materialization.
    feature_spec = self._tf_transform_output.transformed_feature_spec().copy()

    def _pop_labels(features):
      label_key = self.transformed_label_key
      return features, features.pop(label_key)

    def _gzip_reader_fn(files):
      return tf.data.TFRecordDataset(files, compression_type='GZIP')

    def _input_fn() -> tf.data.Dataset:
      dataset = tf.data.experimental.make_batched_features_dataset(
          file_pattern,
          batch_size,
          feature_spec,
          reader=_gzip_reader_fn,
          num_epochs=num_epochs,
          shuffle=shuffle,
          shuffle_buffer_size=shuffle_buffer_size,
          shuffle_seed=shuffle_seed,
          prefetch_buffer_size=prefetch_buffer_size,
          reader_num_threads=reader_num_threads,
          parser_num_threads=parser_num_threads,
          sloppy_ordering=sloppy_ordering,
          drop_final_batch=drop_final_batch)
      if max_num_batches:
        dataset = dataset.take(max_num_batches)
      return dataset.map(_pop_labels)

    return _input_fn

  def get_serving_input_receiver_fn(
      self) -> Callable[[], tf.estimator.export.ServingInputReceiver]:
    """Returns the serving_input_receiver_fn used when exporting a SavedModel.

    Returns:
      An serving_input_receiver_fn. Its returned ServingInputReceiver takes as
      input a batch of raw (i.e. untransformed) serialized tensorflow.Examples.
      The model can be used for serving or for batch inference.
    """

    raw_feature_spec = self._tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(self.raw_label_key)

    def _input_fn() -> tf.estimator.export.ServingInputReceiver:
      raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
          raw_feature_spec, default_batch_size=None)
      serving_input_receiver = raw_input_fn()

      transformed_features = self._tf_transform_output.transform_raw_features(
          serving_input_receiver.features)

      return tf.estimator.export.ServingInputReceiver(
          transformed_features, serving_input_receiver.receiver_tensors)

    return _input_fn

  def get_eval_input_receiver_fn(
      self) -> Callable[[], tfma.export.EvalInputReceiver]:
    """Returns the eval_input_receiver_fn to build a SavedModel for TFMA.

    Returns:
      An eval_input_receiver_fn. Its returned EvalInputReceiver takes as
      input a batch of raw (i.e. untransformed) serialized tensorflow.Examples.
      The model will compute predictions and metrics that TFMA can consume.
    """

    # Notice that the inputs are raw features, not transformed features here.
    raw_feature_spec = self._tf_transform_output.raw_feature_spec()

    def _input_fn() -> tfma.export.EvalInputReceiver:
      serialized_tf_example = tf.compat.v1.placeholder(
          dtype=tf.string, shape=[None], name='input_example_tensor')

      # Add a parse_example operator to the tensorflow graph, which will parse
      # raw, untransformed, tf examples.
      features = tf.io.parse_example(
          serialized_tf_example, features=raw_feature_spec)

      # Now that we have our raw examples, process them through the tf-transform
      # function computed during the preprocessing step.
      transformed_features = self._tf_transform_output.transform_raw_features(
          features)
      label_key = self.transformed_label_key

      # The key name MUST be 'examples'.
      receiver_tensors = {'examples': serialized_tf_example}

      # NOTE: Model is driven by transformed features (since training works on
      # the materialized output of TFT, but slicing will happen on raw features.
      features.update(transformed_features)

      return tfma.export.EvalInputReceiver(
          features=features,
          receiver_tensors=receiver_tensors,
          labels=transformed_features[label_key])

    return _input_fn


def _get_feature_storage_type(schema: schema_pb2.Schema,
                              feature_name: str) -> tf.dtypes.DType:
  """Get the storage type of at tf.Example feature."""

  for feature in schema.feature:
    if feature.name == feature_name:
      if feature.type == schema_pb2.FeatureType.BYTES:
        return tf.string
      if feature.type == schema_pb2.FeatureType.FLOAT:
        return tf.float32
      if feature.type == schema_pb2.FeatureType.INT:
        return tf.int64
  raise ValueError('Feature not found: {}'.format(feature_name))


def _get_feature_dim(schema: schema_pb2.Schema, feature_name: str) -> int:
  """Get the dimension of the tf.Example feature."""

  for feature in schema.feature:
    if feature.name == feature_name:
      return feature.shape.dim[0].size
  raise ValueError('Feature not found: {}'.format(feature_name))
