"""
The module file for Trainer component.
"""
import os
from typing import Dict, List, NamedTuple, Text

import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.utils import io_utils

from datasets import task


def _build_keras_model(feature_spec: Dict[Text, NamedTuple],
                       domains: Dict[Text, NamedTuple],
                       head_size: int,
                       task_type: Text,
                       hidden_units: List[int] = None) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

  Returns:
    A keras Model.
  """

  categorical_columns = []
  real_valued_columns = []

  for key in feature_spec:
    feature_type = feature_spec[key].dtype

    if feature_type == tf.string:
      values = domains[key].value

      feature = tf.feature_column.categorical_column_with_vocabulary_list(
          key, values)
      feature = tf.feature_column.indicator_column(feature)
      categorical_columns.append(feature)

    elif feature_type == tf.int64 or feature_type == tf.int32 or feature_type == tf.int32:
      feature = tf.feature_column.numeric_column(key)
      real_valued_columns.append(feature)

  model = _simple_classifier(
      wide_columns=categorical_columns,
      deep_columns=real_valued_columns,
      feature_spec=feature_spec,
      dnn_hidden_units=hidden_units,
      head_size=head_size,
      task_type=task_type)

  return model


# : List[tf.feature_column.FeatureColumn]
def _simple_classifier(wide_columns,
                       deep_columns,
                       feature_spec,
                       dnn_hidden_units: List[int] = None,
                       head_size: int = 1,
                       task_type: Text = '') -> tf.keras.Model:
  """Builds a simple keras classifier."""

  input_layers = {}
  # Using a simple linear classifier for now.
  for key in feature_spec:

    feature_type = feature_spec[key].dtype

    if feature_type == tf.string:
      input_layers[key] = tf.keras.layers.Input(
          name=key, shape=(), dtype='string')
    elif feature_type == tf.int64 or feature_type == tf.int32 or feature_type == tf.int32:
      keras_tensor = tf.keras.layers.Input(name=key, shape=(), dtype=tf.float32)
      input_layers[key] = keras_tensor

  all_layers = []
  if len(deep_columns) > 0:
    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)

    for numnodes in dnn_hidden_units:
      deep = tf.keras.layers.Dense(numnodes)(deep)

    all_layers.append(deep)

  if len(wide_columns) > 0:
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)
    all_layers.append(wide)

  if len(all_layers) > 1:
    output = tf.keras.layers.concatenate(all_layers)
  else:
    output = all_layers[0]

  if task_type == task.Task.CATEGORICAL_CLASSIFICATION:

    output = tf.keras.layers.Dense(head_size, activation='softmax')(output)
    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

  elif task_type == task.Task.BINARY_CLASSIFICATION:

    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

  elif task_type == task.Task.REGRESSION:
    raise NotImplementedError

  else:
    raise NotImplementedError

  model.summary(print_fn=logging.info)
  return model


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              domains: Dict[Text, NamedTuple],
              label_key: Text,
              batch_size: int = 50) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  output_domain = domains[label_key].value
  initializer = tf.lookup.KeyValueTensorInitializer(
      keys=output_domain,
      values=tf.cast(tf.range(len(output_domain)), tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)

  table = tf.lookup.StaticHashTable(initializer, default_value=-1)

  # check the available feature map
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=label_key)

  def map_fn(x, y, table=table):

    for key in x:
      # x[key] = tf.squeeze(tf.sparse.to_dense(x[key]))
      x[key] = _fill_in_missing(x[key])

    y = table.lookup(tf.squeeze(tf.sparse.to_dense(y)))
    y = tf.one_hot(y, table._initializer._keys.shape[0])
    return (x, y)

  dataset = dataset.map(map_fn)
  return dataset


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  schema_uri = fn_args.schema_file
  schema_proto = io_utils.parse_pbtxt_file(
      file_name=schema_uri, message=schema_pb2.Schema())
  feature_spec, domains = schema_utils.schema_as_feature_spec(schema_proto)

  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 10
  num_dnn_layers = 2
  dnn_decay_factor = 0.5
  label_key = fn_args.label_key
  head_size = fn_args.num_classes
  task_type = fn_args.type

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      tf_transform_output,
      domains=domains,
      label_key=label_key,
      batch_size=40)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      tf_transform_output,
      domains=domains,
      label_key=label_key,
      batch_size=40)

  if label_key in feature_spec:
    feature_spec.pop(label_key)

  model = _build_keras_model(
      # Construct layers sizes with exponetial decay
      feature_spec,
      domains,
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ],
      head_size=head_size,
      task_type=task_type)

  # This log path might change in the future.
  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, tf_transform_output,
                                    label_key).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


def _transformed_name(key):
  return key + '_xf'


def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]


def _get_serve_tf_examples_fn(model, tf_transform_output, label_key):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""

    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(label_key)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)
    transformed_features.pop(label_key)

    for key in transformed_features:
      transformed_features[key] = _fill_in_missing(transformed_features[key])

    return model(transformed_features)

  return serve_tf_examples_fn


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
