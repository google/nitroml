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
from nitroml.components.transform.component import Transform
from nitroml.datasets import openml_cc18
from examples import config
from tfx import components as tfx
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.proto import trainer_pb2
from nitroml.components.meta_learning import meta_learning_wrapper


class OpenMLCC18MetaLearning(nitroml.Benchmark):
  r"""Demos a metalearning pipeline using 'OpenML-CC18' classification datasets."""

  def benchmark(self,
                mock_data: bool = False,
                data_dir: str = None,
                use_keras: bool = True,
                add_publisher: bool = False):

    datasets = openml_cc18.OpenMLCC18(data_dir, mock_data=mock_data)
    dataset_indices = range(len(datasets.names))

    train_stat_gens = []
    train_transforms = []
    test_stat_gens = []
    test_transforms = []
    pipeline = []
    meta_train_data = {}

    if mock_data:
      train_indices = [0, 1]
      test_indices = [0]
    else:
      train_indices = range(21, 26)
      test_indices = range(27, 28)

    for ix, train_index in enumerate(train_indices):

      name = datasets.names[train_index]
      logging.info(f'Train dataset: {name}')
      task_dict = datasets.tasks[train_index].to_dict()
      example_gen = datasets.components[train_index]
      example_gen._instance_name = f'{example_gen._instance_name}.train_{name}'
      stats_gen = tfx.StatisticsGen(
          examples=example_gen.outputs.examples, instance_name=f'train_{name}')
      schema_gen = tfx.SchemaGen(
          statistics=stats_gen.outputs.statistics,
          infer_feature_shape=True,
          instance_name=f'train_{name}')
      transform = Transform(
          examples=example_gen.outputs.examples,
          schema=schema_gen.outputs.schema,
          preprocessing_fn='examples.auto_transform.preprocessing_fn',
          instance_name=f'train_{name}')
      tuner = tfx.Tuner(
          tuner_fn='examples.auto_trainer.tuner_fn',
          examples=transform.outputs.transformed_examples,
          transform_graph=transform.outputs.transform_graph,
          train_args=trainer_pb2.TrainArgs(num_steps=1),
          eval_args=trainer_pb2.EvalArgs(num_steps=1),
          custom_config=task_dict,
          instance_name=f'train_{name}')

      pipeline.extend([example_gen, stats_gen, schema_gen, transform, tuner])

      train_stat_gens.append(stats_gen)
      train_transforms.append(transform)
      meta_train_data[
          f'hparams_train_{ix}'] = tuner.outputs.best_hyperparameters

    meta_learner_helper = meta_learning_wrapper.MetaLearningWrapper(
        train_transformed_examples=train_transforms,
        train_stats_gens=train_stat_gens,
        meta_train_data=meta_train_data)
    pipeline = pipeline + meta_learner_helper.pipeline

    for ix, test_index in enumerate(test_indices):

      name = datasets.names[test_index]
      logging.info(f'Test dataset: {name}')
      task_dict = datasets.tasks[test_index].to_dict()
      example_gen = datasets.components[test_index]
      example_gen._instance_name = f'{example_gen._instance_name}.train_{name}'
      stats_gen = tfx.StatisticsGen(
          examples=example_gen.outputs.examples, instance_name=f'test_{name}')
      schema_gen = tfx.SchemaGen(
          statistics=stats_gen.outputs.statistics,
          infer_feature_shape=True,
          instance_name=f'test_{name}')
      transform = Transform(
          examples=example_gen.outputs.examples,
          schema=schema_gen.outputs.schema,
          preprocessing_fn='examples.auto_transform.preprocessing_fn',
          instance_name=f'test_{name}')
      trainer = tfx.Trainer(
          run_fn='examples.auto_trainer.run_fn',
          custom_executor_spec=(executor_spec.ExecutorClassSpec(
              trainer_executor.GenericExecutor)),
          transformed_examples=transform.outputs.transformed_examples,
          schema=schema_gen.outputs.schema,
          transform_graph=transform.outputs.transform_graph,
          train_args=trainer_pb2.TrainArgs(num_steps=1),
          eval_args=trainer_pb2.EvalArgs(num_steps=1),
          hyperparameters=meta_learner_helper.recommended_search_space,
          custom_config=task_dict)

      # test_stat_gens.append(stats_gen)
      # test_transforms.append(transform)
      pipeline.extend([example_gen, stats_gen, schema_gen, transform, trainer])

      # self.test_without_evaluate(pipeline)
      # Finally, call evaluate() on the workflow DAG outputs, This will
      # automatically append Evaluators to compute metrics from the given
      # SavedModel and 'eval' TF Examples.
      self.evaluate(
          pipeline,
          examples=example_gen.outputs['examples'],
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
        data_dir='/tmp/meta_learning_openML')
