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

from examples import config
from google.protobuf import text_format
from nitroml.components.tuner import component as tuner_component
from nitroml.components.metalearning import metalearning_wrapper
from tfx import components as tfx
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.proto import trainer_pb2
import nitroml


class MetaLearningBenchmark(nitroml.Benchmark):
  r"""Benchmarks a metalearning pipeline on OpenML-CC18 classification tasks."""

  def benchmark(self,
                algorithm: str = None,
                mock_data: bool = False,
                data_dir: str = None):
    # TODO(nikhilmehta): Extend this to multiple test datasets using subbenchmarks.

    train_task_names = frozenset([
        'OpenML.connect4', 'OpenML.creditapproval', 'OpenML.creditg',
        'OpenML.cylinderbands', 'OpenML.diabetes'
    ])
    test_task_names = frozenset(['OpenML.dressessales'])

    if mock_data:
      train_task_names = {'OpenML.mockdata_1'}
      test_task_names = {'OpenML.mockdata_2'}

    train_tasks = []
    test_tasks = []
    for task in nitroml.suites.OpenMLCC18(data_dir, mock_data=mock_data):
      if task.name in train_task_names:
        train_tasks.append(task)
      if task.name in test_task_names:
        test_tasks.append(task)

    pipeline = []
    meta_train_data = {}
    train_autodata_list = []

    for task in train_tasks:

      # Create the autodata instance for this task, which creates Transform,
      # StatisticsGen and SchemaGen component.
      autodata = nitroml.autodata.AutoData(
          task.problem_statement,
          examples=task.train_and_eval_examples,
          preprocessor=nitroml.autodata.BasicPreprocessor(),
          instance_name=f'train_{task.name}')

      # Add a tuner component for each training dataset that finds the optimum HParams.
      tuner = tuner_component.Tuner(
          tuner_fn='examples.auto_trainer.tuner_fn',
          examples=autodata.transformed_examples,
          transform_graph=autodata.transform_graph,
          train_args=trainer_pb2.TrainArgs(num_steps=1),
          eval_args=trainer_pb2.EvalArgs(num_steps=1),
          custom_config={
              # Pass the problem statement proto as a text proto. Required
              # since custom_config must be JSON-serializable.
              'problem_statement':
                  text_format.MessageToString(
                      message=task.problem_statement, as_utf8=True),
          },
          instance_name=f'train_{task.name}')
      pipeline += task.components + autodata.components + [tuner]

      train_autodata_list.append(autodata)
      meta_train_data[
          f'hparams_train_{len(train_autodata_list)}'] = tuner.outputs.best_hyperparameters

    # Construct a MetaLearningHelper that creates the metalearning subpipeline.
    metalearner_helper = metalearning_wrapper.MetaLearningWrapper(
        train_autodata_list=train_autodata_list,
        meta_train_data=meta_train_data)
    pipeline += metalearner_helper.pipeline

    for task in test_tasks:
      with self.sub_benchmark(task.name):
        # Create the autodata instance for the test task.
        autodata = nitroml.autodata.AutoData(
            task.problem_statement,
            examples=task.train_and_eval_examples,
            preprocessor=nitroml.autodata.BasicPreprocessor(),
            instance_name=f'test_{task.name}')

        # Create a trainer component that utilizes the recommended HParams
        # from the metalearning subpipeline.
        trainer = tfx.Trainer(
            run_fn='examples.auto_trainer.run_fn',
            custom_executor_spec=(executor_spec.ExecutorClassSpec(
                trainer_executor.GenericExecutor)),
            transformed_examples=autodata.transformed_examples,
            transform_graph=autodata.transform_graph,
            schema=autodata.schema,
            train_args=trainer_pb2.TrainArgs(num_steps=1),
            eval_args=trainer_pb2.EvalArgs(num_steps=1),
            hyperparameters=metalearner_helper.recommended_search_space,
            custom_config={
                # Pass the problem statement proto as a text proto. Required
                # since custom_config must be JSON-serializable.
                'problem_statement':
                    text_format.MessageToString(
                        message=task.problem_statement, as_utf8=True),
            },
            instance_name=f'test_{task.name}')
        pipeline += task.components + autodata.components + [trainer]

        # Finally, call evaluate() on the workflow DAG outputs, This will
        # automatically append Evaluators to compute metrics from the given
        # SavedModel and 'eval' TF Examples.ss
        self.evaluate(
            pipeline,
            examples=task.train_and_eval_examples,
            model=trainer.outputs.model)


if __name__ == '__main__':

  run_config = dict(
      pipeline_name=config.PIPELINE_NAME + '_metalearning',
      data_dir=config.OTHER_DOWNLOAD_DIR,
      algorithm='majority_voting')

  if config.USE_KUBEFLOW:
    # We need the string "KubeflowDagRunner" in this file to appease the
    # validator used in `tfx create pipeline`.
    # Validator: https://github.com/tensorflow/tfx/blob/v0.22.0/tfx/tools/cli/handler/base_handler.py#L105
    nitroml.main(
        pipeline_root=config.PIPELINE_ROOT,
        tfx_runner=nitroml.get_default_kubeflow_dag_runner(),
        **run_config)
  else:
    # This example has not been tested with engines other than Kubeflow.
    nitroml.main(**run_config)
