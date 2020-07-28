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
# pyformat: disable
r"""Demos a basic NitroML benchmark on the 'Titanic' dataset from OpenML.

To run in open-source:

  python examples/titanic_benchmark.py

"""  # pylint: disable=line-too-long
# pyformat: enable
# pylint: disable=g-import-not-at-top
import os
import sys
# Required since Python binaries ignore relative paths when importing:
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import nitroml
from nitroml.components.tuner import component as custom_tuner
from examples import config
import tensorflow_datasets as tfds
from tfx import components as tfx
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.proto import trainer_pb2

from google.protobuf import text_format


class TitanicBenchmark(nitroml.Benchmark):
  r"""Demos a NitroML benchmark on the 'Titanic' dataset from OpenML."""

  def benchmark(self,
                data_dir: str = None,
                use_keras: bool = True,
                enable_tuning: bool = True):
    # Use TFDSTask to define the task for the titanic dataset.
    task = nitroml.tasks.TFDSTask(tfds.builder('titanic', data_dir=data_dir))

    autodata = nitroml.autodata.AutoData(
        task.problem_statement,
        examples=task.train_and_eval_examples,
        preprocessor=nitroml.autodata.BasicPreprocessor())

    pipeline = task.components + autodata.components

    if enable_tuning:
      # Search over search space of model hyperparameters.
      tuner = custom_tuner.Tuner(
          tuner_fn='examples.auto_trainer.tuner_fn',
          examples=autodata.transformed_examples,
          transform_graph=autodata.transform_graph,
          train_args=trainer_pb2.TrainArgs(num_steps=100),
          eval_args=trainer_pb2.EvalArgs(num_steps=50),
          custom_config={
              # Pass the problem statement proto as a text proto. Required
              # since custom_config must be JSON-serializable.
              'problem_statement':
                  text_format.MessageToString(
                      message=task.problem_statement, as_utf8=True),
          })
      pipeline.append(tuner)

    # Define a Trainer to train our model on the given task.
    trainer = tfx.Trainer(
        run_fn='examples.auto_trainer.run_fn'
        if use_keras else 'examples.auto_estimator_trainer.run_fn',
        custom_executor_spec=(executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor)),
        transformed_examples=autodata.transformed_examples,
        transform_graph=autodata.transform_graph,
        schema=autodata.schema,
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500),
        hyperparameters=(tuner.outputs.best_hyperparameters
                         if enable_tuning else None),
        custom_config={
            # Pass the problem statement proto as a text proto. Required
            # since custom_config must be JSON-serializable.
            'problem_statement':
                text_format.MessageToString(
                    message=task.problem_statement, as_utf8=True),
        })

    pipeline.append(trainer)

    # Finally, call evaluate() on the workflow DAG outputs. This will
    # automatically append Evaluators to compute metrics from the given
    # SavedModel and 'eval' TF Examples.
    self.evaluate(
        pipeline,
        examples=task.train_and_eval_examples,
        model=trainer.outputs.model)


if __name__ == '__main__':
  if config.USE_KUBEFLOW:
    # We need the string "KubeflowDagRunner" in this file to appease the
    # validator used in `tfx create pipeline`.
    # Validator: https://github.com/tensorflow/tfx/blob/v0.22.0/tfx/tools/cli/handler/base_handler.py#L105
    nitroml.main(
        pipeline_name=config.PIPELINE_NAME + '_titanic',
        pipeline_root=config.PIPELINE_ROOT,
        data_dir=config.TF_DOWNLOAD_DIR,
        tfx_runner=nitroml.get_default_kubeflow_dag_runner())
  else:
    nitroml.main()
