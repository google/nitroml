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
r"""Demos a basic NitroML benchmark on 'OpenMLCC18' datasets from OpenML.

To run in open-source:

  python examples/openml_cc18_benchmark.py

"""  # pylint: disable=line-too-long
# pyformat: enable
# pylint: disable=g-import-not-at-top
import os
import sys
import json
# Required since Python binaries ignore relative paths when importing:
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from absl import logging
import nitroml
from examples import config
from tfx import components as tfx
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.proto import trainer_pb2
from tfx.components.base import base_component
from nitroml.components.meta_learning import meta_learning_wrapper
from nitroml.components.meta_learning import META_LEARNING_ALGORITHMS

from google.protobuf import text_format


class OpenMLCC18MetaLearning(nitroml.Benchmark):
  r"""Demos a metalearning pipeline using 'OpenML-CC18' classification datasets."""

  def set_instance_name(self,
                        component: base_component.BaseComponent,
                        suffix: str = ''):
    logging.info(component._instance_name)
    if component._instance_name:
      component._instance_name = f'{component._instance_name}.{suffix}'
    else:
      component._instance_name = suffix

  def benchmark(self,
                algorithm: str = None,
                mock_data: bool = False,
                data_dir: str = None):

    if not algorithm or algorithm not in META_LEARNING_ALGORITHMS:
      raise ValueError(
          f'Required a valid meta learning algorithm. Found "{algorithm}", Expected one of {META_LEARNING_ALGORITHMS}'
      )

    pipeline = []
    meta_train_data = {}
    train_autodata_list = []

    # TODO: Think of a better way to create this experiment,
    # Create train/test datasets.
    if mock_data:
      train_indices = [0, 1]
      test_indices = [0]
    else:
      train_indices = range(21, 26)
      test_indices = range(27, 28)

    pipeline = []
    for train_index, task in enumerate(
        nitroml.suites.OpenMLCC18(data_dir, mock_data=mock_data)):

      if train_index not in train_indices:
        continue

      instance_name = f'train_{task.name}'
      autodata = nitroml.autodata.AutoData(
          task.problem_statement,
          examples=task.train_and_eval_examples,
          preprocessor=nitroml.autodata.BasicPreprocessor(),
          instance_name=instance_name)

      self.set_instance_name(task.components[0], instance_name)
      pipeline += task.components + autodata.components

      tuner = tfx.Tuner(
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
      pipeline.append(tuner)

      train_autodata_list.append(autodata)
      meta_train_data[
          f'hparams_train_{len(train_autodata_list)}'] = tuner.outputs.best_hyperparameters

    meta_learner_helper = meta_learning_wrapper.MetaLearningWrapper(
        train_autodata_list=train_autodata_list,
        meta_train_data=meta_train_data)
    pipeline += pipeline + meta_learner_helper.pipeline

    for test_index, task in enumerate(
        nitroml.suites.OpenMLCC18(data_dir, mock_data=mock_data)):

      if test_index not in test_indices:
        continue

      instance_name = f'test_{task.name}'
      autodata = nitroml.autodata.AutoData(
          task.problem_statement,
          examples=task.train_and_eval_examples,
          preprocessor=nitroml.autodata.BasicPreprocessor(),
          instance_name=instance_name)

      self.set_instance_name(task.components[0], instance_name)
      pipeline += task.components + autodata.components

      trainer = tfx.Trainer(
          run_fn='examples.auto_trainer.run_fn',
          custom_executor_spec=(executor_spec.ExecutorClassSpec(
              trainer_executor.GenericExecutor)),
          transformed_examples=autodata.transformed_examples,
          transform_graph=autodata.transform_graph,
          schema=autodata.schema,
          train_args=trainer_pb2.TrainArgs(num_steps=1),
          eval_args=trainer_pb2.EvalArgs(num_steps=1),
          hyperparameters=meta_learner_helper.recommended_search_space,
          custom_config={
              # Pass the problem statement proto as a text proto. Required
              # since custom_config must be JSON-serializable.
              'problem_statement':
                  text_format.MessageToString(
                      message=task.problem_statement, as_utf8=True),
          })
      pipeline.append(trainer)

      # Finally, call evaluate() on the workflow DAG outputs, This will
      # automatically append Evaluators to compute metrics from the given
      # SavedModel and 'eval' TF Examples.ss
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
        pipeline_name=config.PIPELINE_NAME + '_metalearning',
        pipeline_root=config.PIPELINE_ROOT,
        data_dir=config.OTHER_DOWNLOAD_DIR,
        tfx_runner=nitroml.get_default_kubeflow_dag_runner())
  else:
    # This example has not been tested with engines other than Kubeflow.
    nitroml.main(
        pipeline_name=config.PIPELINE_NAME + '_metalearning',
        data_dir=config.OTHER_DOWNLOAD_DIR,
    )
