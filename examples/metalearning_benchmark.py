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
"""Demos a metalearning pipeline as a NitroML benchmark on 'OpenMLCC18' datasets.

To run in open-source:

  python -m examples.metalearning_benchmark.py

"""  # pylint: disable=line-too-long
# pyformat: enable
# pylint: disable=g-import-not-at-top
import os
import sys

# Required since Python binaries ignore relative paths when importing:
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import nitroml
from nitroml.automl import autodata as ad
from nitroml.automl.metalearning import metalearning as mtl
from nitroml.automl.metalearning.tuner import component as tuner_component
from nitroml.benchmark.suites import openml_cc18
from examples import config
from tfx import components as tfx
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.base import executor_spec
from tfx.proto import trainer_pb2

from google.protobuf import text_format


class MetaLearningBenchmark(nitroml.Benchmark):
  r"""Benchmarks a metalearning pipeline on OpenML-CC18 classification tasks."""

  def benchmark(self,
                algorithm: str = None,
                mock_data: bool = False,
                data_dir: str = None):
    # TODO(nikhilmehta): Extend this to multiple test datasets using subbenchmarks.

    metatrain_task_names = frozenset([
        'OpenML.connect4', 'OpenML.creditapproval', 'OpenML.creditg',
        'OpenML.cylinderbands', 'OpenML.diabetes'
    ])
    metatest_task_names = frozenset(['OpenML.dressessales'])
    train_steps = 1000

    if mock_data:
      metatrain_task_names = {'OpenML.mockdata_1'}
      metatest_task_names = {'OpenML.mockdata_2'}
      train_steps = 10

    metatrain_tasks = []
    metatest_tasks = []
    for task in openml_cc18.OpenMLCC18(data_dir, mock_data=mock_data):
      if task.name in metatrain_task_names:
        metatrain_tasks.append(task)
      if task.name in metatest_task_names:
        metatest_tasks.append(task)

    meta_train_data = {}
    train_autodata_list = []
    for task in metatrain_tasks:
      # Register running the Task's data preparation components.
      self.add(task.components)

      # Create the autodata instance for this task, which creates Transform,
      # StatisticsGen and SchemaGen component.
      autodata = self.add(
          ad.AutoData(
              task.problem_statement,
              examples=task.train_and_eval_examples,
              preprocessor=ad.BasicPreprocessor(),
              instance_name=f'train.{task.name}'))

      # Add a tuner component for each metatrain dataset that finds the optimum
      # HParams.
      tuner = self.add(
          tuner_component.AugmentedTuner(
              tuner_fn='nitroml.automl.autotrainer.lib.auto_trainer.tuner_fn',
              examples=autodata.outputs.transformed_examples,
              transform_graph=autodata.outputs.transform_graph,
              train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
              eval_args=trainer_pb2.EvalArgs(num_steps=1),
              custom_config={
                  # Pass the problem statement proto as a text proto. Required
                  # since custom_config must be JSON-serializable.
                  'problem_statement':
                      text_format.MessageToString(
                          message=task.problem_statement, as_utf8=True),
              },
              instance_name=f'train.{task.name}'))
      train_autodata_list.append(autodata)
      key = f'hparams_train_{len(train_autodata_list)}'
      meta_train_data[key] = tuner.outputs.best_hyperparameters

    # Construct the MetaLearning subpipeline.
    metalearning = self.add(
        mtl.MetaLearning(
            train_autodata_list=train_autodata_list,
            meta_train_data=meta_train_data,
            algorithm=algorithm))

    for task in metatest_tasks:
      with self.sub_benchmark(task.name):
        # Register running the Task's data preparation components.
        self.add(task.components)

        # Create the autodata instance for the test task.
        autodata = self.add(
            ad.AutoData(
                task.problem_statement,
                examples=task.train_and_eval_examples,
                preprocessor=ad.BasicPreprocessor()))

        test_meta_components, best_hparams = metalearning.create_test_components(
            autodata, tuner_steps=train_steps)
        self.add(test_meta_components)

        # Create a trainer component that utilizes the recommended HParams
        # from the metalearning subpipeline.
        trainer = self.add(
            tfx.Trainer(
                run_fn='nitroml.automl.autotrainer.lib.auto_trainer.run_fn',
                custom_executor_spec=(executor_spec.ExecutorClassSpec(
                    trainer_executor.GenericExecutor)),
                transformed_examples=autodata.outputs.transformed_examples,
                transform_graph=autodata.outputs.transform_graph,
                schema=autodata.outputs.schema,
                train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
                eval_args=trainer_pb2.EvalArgs(num_steps=1),
                hyperparameters=best_hparams,
                custom_config={
                    # Pass the problem statement proto as a text proto. Required
                    # since custom_config must be JSON-serializable.
                    'problem_statement':
                        text_format.MessageToString(
                            message=task.problem_statement, as_utf8=True),
                }))

        # Finally, call evaluate() on the workflow DAG outputs, This will
        # automatically append Evaluators to compute metrics from the given
        # SavedModel and 'eval' TF Examples.ss
        self.evaluate(task=task, model=trainer.outputs.model)


if __name__ == '__main__':

  metalearning_algorithm = 'nearest_neighbor'
  run_config = dict(
      pipeline_name=f'metalearning_{metalearning_algorithm}',
      data_dir=config.OTHER_DOWNLOAD_DIR,
      algorithm=metalearning_algorithm)

  if config.USE_KUBEFLOW:
    # We need the string "KubeflowDagRunner" in this file to appease the
    # validator used in `tfx create pipeline`.
    # Validator: https://github.com/tensorflow/tfx/blob/v0.22.0/tfx/tools/cli/handler/base_handler.py#L105
    nitroml.main(
        pipeline_root=os.path.join(config.PIPELINE_ROOT,
                                   run_config['pipeline_name']),
        tfx_runner=nitroml.get_default_kubeflow_dag_runner(),
        **run_config)
  else:
    # This example has not been tested with engines other than Kubeflow.
    nitroml.main(**run_config)
