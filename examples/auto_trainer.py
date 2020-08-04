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
"""Auto Keras trainer that trains on any tabular dataset.

The consumed artifacts include:
 * Dataset schema.
 * TensorFlow Transform outputs.
"""

from typing import Any, Callable, List, Optional, Dict

from absl import logging
import kerastuner
from kerastuner.engine import hyperparameters as hp_module
from nitroml.components.tuner.executor import get_tuner_cls_with_callbacks
from nitroml.protos import problem_statement_pb2 as ps_pb2
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tfx.components.trainer import executor as trainer_executor
from tfx.components.trainer import fn_args_utils
from tfx.components.tuner.component import TunerFnResult

from google.protobuf import text_format

FeatureColumn = Any


def _get_hyperparameters() -> kerastuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""

  hp = kerastuner.HyperParameters()
  hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3], default=1e-2)
  hp.Choice('optimizer', ['Adam', 'SGD', 'RMSprop', 'Adagrad'], default='Adam')
  hp.Int('num_layers', min_value=1, max_value=5, step=1, default=2)
  hp.Int('num_nodes', min_value=32, max_value=512, step=32, default=128)
  return hp


def tuner_fn(fn_args: fn_args_utils.FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.

  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
      - custom_config: A dict with a single 'problem_statement' entry containing
        a text-format serialized ProblemStatement proto which defines the task.

  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """

  # TODO(weill): Replace with AutoDataProvider.
  data_provider = KerasDataProvider(
      problem_statement=text_format.Parse(
          fn_args.custom_config['problem_statement'],
          ps_pb2.ProblemStatement()),
      transform_graph_dir=fn_args.transform_graph_path)

  build_keras_model = lambda hparams: _build_keras_model(data_provider, hparams)
  tuner_cls = get_tuner_cls_with_callbacks(kerastuner.RandomSearch)
  tuner = tuner_cls(
      build_keras_model,
      max_trials=fn_args.custom_config.get('max_trials', 10),
      hyperparameters=(hp_module.HyperParameters.from_config(
          fn_args.custom_config.get('warmup_hyperparameters'))
                       if 'warmup_hyperparameters' in fn_args.custom_config else
                       _get_hyperparameters()),
      allow_new_entries=False,
      objective=data_provider.tuner_objective,
      directory=fn_args.working_dir,
      project_name=f'{data_provider.task_name}_tuning')

  train_dataset = data_provider.get_input_fn(
      file_pattern=fn_args.train_files,
      batch_size=64,
      num_epochs=None,
      shuffle=True)()

  eval_dataset = data_provider.get_input_fn(
      file_pattern=fn_args.eval_files,
      batch_size=64,
      num_epochs=1,
      shuffle=False)()

  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
      })


def run_fn(fn_args: trainer_executor.TrainerFnArgs):
  """Train a DNN Keras Model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
      - train_files: A list of uris for train files.
      - transform_output: An optional single uri for transform graph produced by
        TFT. Will be None if not specified.
      - serving_model_dir: A single uri for the output directory of the serving
        model.
      - eval_model_dir: A single uri for the output directory of the eval model.
        Note that this is for estimator only, Keras doesn't require it for TFMA.
      - eval_files:  A list of uris for eval files.
      - schema_file: A single uri for schema file.
      - train_steps: Number of train steps.
      - eval_steps: Number of eval steps.
      - base_model: Base model that will be used for this training job.
      - hyperparameters: An optional kerastuner.HyperParameters config.
      - problem_statement: A text-format serialized ProblemStatement proto
        which defines the task.
  """

  # TODO(weill): Replace with AutoDataProvider.
  data_provider = KerasDataProvider(
      problem_statement=text_format.Parse(fn_args.problem_statement,
                                          ps_pb2.ProblemStatement()),
      transform_graph_dir=fn_args.transform_output)

  if fn_args.hyperparameters:
    hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    hparams = _get_hyperparameters()
  logging.info('HyperParameters for training: %s', hparams.get_config())
  model = _build_keras_model(data_provider, hparams)

  train_spec = tf.estimator.TrainSpec(
      input_fn=data_provider.get_input_fn(
          file_pattern=fn_args.train_files,
          batch_size=64,
          num_epochs=None,
          shuffle=True),
      max_steps=fn_args.train_steps)

  serving_receiver_fn = data_provider.get_serving_input_receiver_fn()
  exporters = [
      tf.estimator.FinalExporter('serving_model_dir', serving_receiver_fn),
  ]
  eval_spec = tf.estimator.EvalSpec(
      input_fn=data_provider.get_input_fn(
          file_pattern=fn_args.eval_files,
          batch_size=64,
          num_epochs=1,
          shuffle=False),
      steps=fn_args.eval_steps,
      exporters=exporters,
      # Since eval runs in parallel, we can begin evaluation as soon as new
      # checkpoints are written.
      start_delay_secs=1,
      throttle_secs=5)

  run_config = tf.estimator.RunConfig(
      model_dir=fn_args.serving_model_dir,
      save_checkpoints_steps=999,
      keep_checkpoint_max=3)

  estimator = tf.keras.estimator.model_to_estimator(model, config=run_config)
  logging.info('Training model...')
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  logging.info('Training complete.')

  # Export an eval savedmodel for TFMA. If distributed training, it must only
  # be written by the chief worker, as would be done for serving savedmodel.
  if run_config.is_chief:
    logging.info('Exporting eval_savedmodel for TFMA.')
    tfma.export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=fn_args.eval_model_dir,
        eval_input_receiver_fn=data_provider.get_eval_input_receiver_fn())

    logging.info('Exported eval_savedmodel to %s.', fn_args.eval_model_dir)
  else:
    logging.info('eval_savedmodel export for TFMA is skipped because '
                 'this is not the chief worker.')


# TODO(weill): Replace with AutoData
class KerasDataProvider:
  """Creates feature columns and specs from TFX artifacts."""

  SPARSE_CATEGORICAL_CE = 'sparse_categorical_crossentropy'
  CATEGORICAL_CE = 'categorical_crossentropy'
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

  @property
  def label_key(self) -> str:
    task_type = self._problem_statement.tasks[0].type
    if task_type.HasField('multi_class_classification'):
      return task_type.multi_class_classification.label
    if task_type.HasField('binary_classification'):
      return task_type.binary_classification.label
    if task_type.HasField('one_dimensional_regression'):
      return task_type.one_dimensional_regression.label
    raise ValueError('Invalid task type: {}'.format(task_type))

  @property
  def num_classes(self) -> int:
    # Subtract `num_oov_buckets` used in base_processor.py
    return self._tf_transform_output.num_buckets_for_transformed_feature(
        self.label_key) - 1

  @property
  def task_name(self) -> str:
    return self._problem_statement.tasks[0].name

  @property
  def raw_label_keys(self) -> List[str]:
    """The raw label key as defined in the ProblemStatement."""

    # TODO(nikhilmehta): Change the task object to allow label_key to be a list.
    return [self.label_key]

  @property
  def transformed_label_keys(self) -> List[str]:
    """The label key after applying TensorFlow Transform to the Examples."""

    return self.raw_label_keys

  @property
  def head_size(self) -> int:
    """Returns the head size for this task."""

    # TODO(github.com/googleinterns/nitroml/issues/29): Regression tasks
    # (self.num_classes==0)
    if self.num_classes == 2:
      return 1
    return self.num_classes

  @property
  def loss(self) -> str:
    """Returns the keras loss_fn for this task."""

    if self.num_classes > 2:
      return self.SPARSE_CATEGORICAL_CE
    return self.BINARY_CE

  @property
  def head_activation(self) -> str:
    """Returns the activation for the final layer."""

    if self.num_classes > 2:
      return 'softmax'
    return 'sigmoid'

  @property
  def metrics(self) -> List[tf.keras.metrics.Metric]:

    if self.num_classes == 2:
      return [
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.BinaryCrossentropy(name='average_loss'),
          tf.keras.metrics.AUC()
      ]
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseCategoricalCrossentropy(name='average_loss')
    ]

  @property
  def tuner_objective(self) -> kerastuner.Objective:

    if self.num_classes == 2:
      return kerastuner.Objective('val_auc', 'max')
    else:
      return kerastuner.Objective('val_accuracy', 'max')

  def get_input_layers(self) -> Dict[str, tf.keras.layers.Input]:
    """Returns input layers for a Keras Model."""

    feature_spec = self._tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(self.transformed_label_keys[0])
    input_layers = {}
    for colname, spec in feature_spec.items():
      input_layers[colname] = tf.keras.layers.Input(
          name=colname, shape=spec.shape, dtype=spec.dtype)
    return input_layers

  def get_numeric_feature_columns(self,
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
      if feature_name in self.raw_label_keys:
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

  def get_sparse_categorical_feature_columns(
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
      if feature_name in self.raw_label_keys:
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
      else:
        # Note TFT automatically converts string columns to int columns.
        raise ValueError('Unsupported dtype.')
      feature_columns.append(new_feature_column)
    return feature_columns

  def get_embedding_feature_columns(self,
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
        self.get_sparse_categorical_feature_columns(include_integer_columns)
    ]

  def get_serving_input_receiver_fn(
      self) -> Callable[[], tf.estimator.export.ServingInputReceiver]:
    """Returns the serving_input_receiver_fn used when exporting a SavedModel.

    Returns:
      An serving_input_receiver_fn. Its returned ServingInputReceiver takes as
      input a batch of raw (i.e. untransformed) serialized tensorflow.Examples.
      The model can be used for serving or for batch inference.
    """

    raw_feature_spec = self._tf_transform_output.raw_feature_spec()
    for key in self.raw_label_keys:
      raw_feature_spec.pop(key)

    def _input_fn() -> tf.estimator.export.ServingInputReceiver:
      raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
          raw_feature_spec, default_batch_size=None)
      serving_input_receiver = raw_input_fn()

      transformed_features = self._tf_transform_output.transform_raw_features(
          serving_input_receiver.features)

      transformed_features.pop(self.raw_label_keys[0])

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

      # The key name MUST be 'examples'.
      receiver_tensors = {'examples': serialized_tf_example}

      # NOTE: Model is driven by transformed features (since training works on
      # the materialized output of TFT, but slicing will happen on raw features.
      features.update(transformed_features)

      def _pop_labels(features):
        label_keys = self.transformed_label_keys
        labels = []
        for key in label_keys:
          labels.append(features.pop(key))
        return tf.concat(labels, axis=1)

      labels = _pop_labels(features)
      return tfma.export.EvalInputReceiver(
          features=features, receiver_tensors=receiver_tensors, labels=labels)

    return _input_fn

  def get_input_fn(
      self,
      file_pattern: List[str],
      batch_size: int,
      num_epochs: Optional[int] = None,
      shuffle: Optional[bool] = True,
      shuffle_buffer_size: int = 10000,
      shuffle_seed: Optional[int] = None,
      prefetch_buffer_size: Optional[int] = None,
      reader_num_threads: Optional[int] = None,
      parser_num_threads: Optional[int] = None,
      sloppy_ordering: bool = False,
      drop_final_batch: bool = False) -> Callable[[], tf.data.Dataset]:
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
      Returns an input_fn that returns a `tf.data.Dataset`.
    """

    # Since we're not applying the transform graph here, we're using Transform
    # materialization.
    feature_spec = self._tf_transform_output.transformed_feature_spec().copy()

    def _pop_labels(features):
      label_keys = self.transformed_label_keys
      labels = []
      for key in label_keys:
        labels.append(features.pop(key))
      return features, tf.concat(labels, axis=1)

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
      return dataset.map(_pop_labels)

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


def _build_keras_model(data_provider: KerasDataProvider,
                       hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Returns a Keras Model for the given data adapter.

  Args:
    data_provider: Data adaptor used to get the task information.
    hparams: Hyperparameters of the model.

  Returns:
    A keras model for the given adapter and hyperparams.
  """

  feature_columns = data_provider.get_numeric_feature_columns(
  ) + data_provider.get_embedding_feature_columns()
  input_layers = data_provider.get_input_layers()

  # All input_layers must be consumed for the Keras Model to work.
  assert len(feature_columns) >= len(input_layers)

  x = tf.keras.layers.DenseFeatures(feature_columns)(input_layers)

  hparam_nodes = hparams.get('num_nodes')
  for numnodes in [hparam_nodes] * hparams.get('num_layers'):
    x = tf.keras.layers.Dense(numnodes)(x)
  output = tf.keras.layers.Dense(
      data_provider.head_size,
      activation=data_provider.head_activation,
      name='output')(
          x)

  model = tf.keras.Model(input_layers, output)

  lr = float(hparams.get('learning_rate'))
  optimizer_str = hparams.get('optimizer')
  if optimizer_str == 'Adam':
    optimizer = tf.keras.optimizers.Adam(lr=lr)
  elif optimizer_str == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(lr=lr)
  elif optimizer_str == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(lr=lr)
  elif optimizer_str == 'SGD':
    optimizer = tf.keras.optimizers.SGD(lr=lr)

  model.compile(
      loss=data_provider.loss,
      optimizer=optimizer,
      metrics=data_provider.metrics)
  model.summary()

  return model
