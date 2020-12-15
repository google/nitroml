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

import functools
from typing import Optional

from absl import logging
import kerastuner
from kerastuner.engine import hyperparameters as hp_module
from nitroml.automl.autodata.trainer_adapters import estimator_adapter as ea
from nitroml.automl.autodata.trainer_adapters import keras_model_adapter as kma
from nitroml.automl.metalearning.tuner.executor import get_tuner_cls_with_callbacks
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components.trainer import executor as trainer_executor
from tfx.components.trainer import fn_args_utils
from tfx.components.tuner.component import TunerFnResult

from google.protobuf import text_format
from nitroml.protos import problem_statement_pb2 as ps_pb2


def _get_hyperparameters() -> kerastuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""

  hp = kerastuner.HyperParameters()
  hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4], default=1e-3)
  hp.Choice('optimizer', ['Adam', 'SGD', 'RMSprop', 'Adagrad'], default='Adam')
  hp.Int('num_layers', min_value=1, max_value=5, step=1, default=1)
  hp.Int('num_nodes', min_value=32, max_value=512, step=32, default=64)
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

  problem_statement = text_format.Parse(
      fn_args.custom_config['problem_statement'], ps_pb2.ProblemStatement())
  autodata_adapter = kma.KerasModelAdapter(
      problem_statement=problem_statement,
      transform_graph_dir=fn_args.transform_graph_path)

  build_keras_model_fn = functools.partial(
      _build_keras_model, autodata_adapter=autodata_adapter)
  if 'warmup_hyperparameters' in fn_args.custom_config:
    hyperparameters = hp_module.HyperParameters.from_config(
        fn_args.custom_config['warmup_hyperparameters'])
  else:
    hyperparameters = _get_hyperparameters()

  tuner_cls = get_tuner_cls_with_callbacks(kerastuner.RandomSearch)
  tuner = tuner_cls(
      build_keras_model_fn,
      max_trials=fn_args.custom_config.get('max_trials', 10),
      hyperparameters=hyperparameters,
      allow_new_entries=False,
      objective=autodata_adapter.tuner_objective,
      directory=fn_args.working_dir,
      project_name=f'{problem_statement.tasks[0].name}_tuning')

  # TODO(nikhilmehta): Make batch-size tunable hyperparameter.
  train_dataset = autodata_adapter.get_dataset(
      file_pattern=fn_args.train_files,
      batch_size=128,
      num_epochs=None,
      shuffle=True)

  eval_dataset = autodata_adapter.get_dataset(
      file_pattern=fn_args.eval_files,
      batch_size=128,
      num_epochs=1,
      shuffle=False)

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
      - custom_config: A dict with a single 'problem_statement' entry containing
        a text-format serialized ProblemStatement proto which defines the task.
  """

  # Use EstimatorAdapter here because we will wrap the Keras Model into an
  # Estimator for training and export.
  autodata_adapter = ea.EstimatorAdapter(
      problem_statement=text_format.Parse(
          fn_args.custom_config['problem_statement'],
          ps_pb2.ProblemStatement()),
      transform_graph_dir=fn_args.transform_output)

  if fn_args.hyperparameters:
    hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    hparams = _get_hyperparameters()
  logging.info('HyperParameters for training: %s', hparams.get_config())

  # Use KerasAdapter here because we create need it to create the Keras Model.
  keras_autodata_adapter = kma.KerasModelAdapter(
      problem_statement=text_format.Parse(
          fn_args.custom_config['problem_statement'],
          ps_pb2.ProblemStatement()),
      transform_graph_dir=fn_args.transform_output)
  model = _build_keras_model(
      hparams,
      keras_autodata_adapter,
      sequence_length=fn_args.custom_config.get('sequence_length', None))

  train_spec = tf.estimator.TrainSpec(
      input_fn=autodata_adapter.get_input_fn(
          file_pattern=fn_args.train_files,
          batch_size=64,
          num_epochs=None,
          shuffle=True),
      max_steps=fn_args.train_steps)

  serving_receiver_fn = autodata_adapter.get_serving_input_receiver_fn()
  exporters = [
      tf.estimator.FinalExporter('serving_model_dir', serving_receiver_fn),
  ]
  eval_spec = tf.estimator.EvalSpec(
      input_fn=autodata_adapter.get_input_fn(
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
        eval_input_receiver_fn=autodata_adapter.get_eval_input_receiver_fn())

    logging.info('Exported eval_savedmodel to %s.', fn_args.eval_model_dir)
  else:
    logging.info('eval_savedmodel export for TFMA is skipped because '
                 'this is not the chief worker.')


def _build_keras_model(hparams: kerastuner.HyperParameters,
                       autodata_adapter: kma.KerasModelAdapter,
                       sequence_length: Optional[int] = None) -> tf.keras.Model:
  """Returns a Keras Model for the given data adapter.

  Args:
    hparams: Hyperparameters of the model.
    autodata_adapter: Data adaptor used to get the task information.
    sequence_length: The length of the sequence to predict when not-None.

  Returns:
    A keras model for the given adapter and hyperparams.
  """

  feature_columns = autodata_adapter.get_dense_feature_columns()
  input_layers = autodata_adapter.get_input_layers()

  # All input_layers must be consumed for the Keras Model to work.
  assert len(feature_columns) >= len(input_layers)

  x = tf.keras.layers.DenseFeatures(feature_columns)(input_layers)

  num_nodes = hparams.get('num_nodes')
  if sequence_length:
    logging.info('Creating an LSTM model with prediction sequence length: %s.',
                 sequence_length)

    x = tf.expand_dims(x, axis=1)
    x = tf.keras.layers.LSTM(
        num_nodes, activation='relu', input_shape=(1, None))(
            x)
    # repeat vector
    x = tf.keras.layers.RepeatVector(sequence_length)(x)
    # decoder layer
    x = tf.keras.layers.LSTM(
        num_nodes, activation='relu', return_sequences=True)(
            x)
    output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            autodata_adapter.head_size,
            activation=autodata_adapter.head_activation),
        name='output')(
            x)
  else:
    logging.info('Creating an densely-connected DNN model.')

    for numnodes in [num_nodes] * hparams.get('num_layers'):
      x = tf.keras.layers.Dense(numnodes)(x)
    output = tf.keras.layers.Dense(
        autodata_adapter.head_size,
        activation=autodata_adapter.head_activation,
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
      loss=autodata_adapter.loss,
      optimizer=optimizer,
      metrics=autodata_adapter.metrics)
  model.summary()

  return model
