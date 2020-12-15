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

from typing import Any

from absl import logging
from nitroml.automl.autodata.trainer_adapters import estimator_adapter
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components.trainer import executor as trainer_executor

from google.protobuf import text_format
from nitroml.protos import problem_statement_pb2 as ps_pb2

FeatureColumn = Any


# TFX Trainer will call this function.
def run_fn(fn_args: trainer_executor.TrainerFnArgs):
  """Train a DNNEstimator based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
      - train_files: A list of uris for train files.
      - transform_output: An optional single uri for transform graph produced by
        TFT. Will be None if not specified.
      - serving_model_dir: A single uri for the output directory of the serving
        model.
      - eval_model_dir: A single uri for the output directory of the eval model.
        Note that this is estimator only, Keras doesn't require it for TFMA.
      - eval_files:  A list of uris for eval files.
      - schema_file: A single uri for schema file.
      - train_steps: Number of train steps.
      - eval_steps: Number of eval steps.
      - base_model: Base model that will be used for this training job.
      - hyperparameters: An optional kerastuner.HyperParameters config.
      - custom_config: A dict with a single 'problem_statement' entry containing
        a text-format serialized ProblemStatement proto which defines the task.
  """
  sequence_length = fn_args.custom_config.get('sequence_length', None)
  if sequence_length:
    raise ValueError('Sequential prediction tasks are not supported. '
                     'Set `use_keras=True` in AutoTrainer instead.')

  autodata_adapter = estimator_adapter.EstimatorAdapter(
      problem_statement=text_format.Parse(
          fn_args.custom_config['problem_statement'],
          ps_pb2.ProblemStatement()),
      transform_graph_dir=fn_args.transform_output)

  run_config = tf.estimator.RunConfig(
      model_dir=fn_args.serving_model_dir,
      save_checkpoints_steps=999,
      keep_checkpoint_max=3)

  estimator = tf.estimator.DNNEstimator(
      head=autodata_adapter.head,
      feature_columns=autodata_adapter.get_dense_feature_columns(),
      hidden_units=[128, 128],
      config=run_config)

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

  # Train/Tune the model
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
