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
                enable_tuning: bool = True,
                add_publisher: bool = False):

    datasets = openml_cc18.OpenMLCC18(data_dir, mock_data=mock_data)
    dataset_indices = range(len(datasets.names))
    # meta_datasets = ['adult', 'car', 'segment']
    meta_datasets = [
        'mockdata',
        'mockdata',
    ]
    meta_train_datasets = meta_datasets[:1]
    meta_test_datasets = meta_datasets[1:2]

    train_stat_gens = []
    train_transforms = []
    test_stat_gens = []
    test_transforms = []
    pipeline = []

    # TODO(nikhilmehta: Add instance_name)
    for ix in [0, 1]:

      name = datasets.names[0]
      task_dict = datasets.tasks[0].to_dict()

      if name.lower() not in meta_datasets:
        logging.info('Skipping %s', name)
        continue

      example_gen = datasets.components[0]
      pipeline.append(example_gen)

      stats_gen = tfx.StatisticsGen(
          examples=example_gen.outputs.examples, instance_name=name + f'_{ix}')
      schema_gen = tfx.SchemaGen(
          statistics=stats_gen.outputs.statistics,
          infer_feature_shape=True,
          instance_name=name + f'_{ix}')
      transform = Transform(
          examples=example_gen.outputs.examples,
          schema=schema_gen.outputs.schema,
          preprocessing_fn='examples.auto_transform.preprocessing_fn',
          instance_name=name + f'_{ix}')
      pipeline.extend([stats_gen, schema_gen, transform])

      #TODO(nikhilmehta): Remove the ix == 0 check.
      if name.lower() in meta_train_datasets and ix == 0:
        train_stat_gens.append(stats_gen)
        train_transforms.append(transform)
      else:
        test_stat_gens.append(stats_gen)
        test_transforms.append(transform)

    meta_learner_helper = meta_learning_wrapper.MetaLearningWrapper(
        train_transformed_examples=train_transforms,
        train_stats_gens=train_stat_gens,
        test_transformed_examples=test_transforms,
        test_stats_gens=test_stat_gens)

    pipeline = pipeline + meta_learner_helper.pipeline

    self.test_without_evaluate(pipeline)


if __name__ == '__main__':

  if config.USE_KUBEFLOW:
    # We need the string "KubeflowDagRunner" in this file to appease the
    # validator used in `tfx create pipeline`.
    # Validator: https://github.com/tensorflow/tfx/blob/v0.22.0/tfx/tools/cli/handler/base_handler.py#L105
    nitroml.main(
        pipeline_name=config.PIPELINE_NAME + '_openML',
        pipeline_root=config.PIPELINE_ROOT,
        data_dir=config.OTHER_DOWNLOAD_DIR,
        tfx_runner=nitroml.get_default_kubeflow_dag_runner())
  else:
    # This example has not been tested with engines other than Kubeflow.
    nitroml.main(
        pipeline_name=config.PIPELINE_NAME + '_meta_learning',
        data_dir='/tmp/meta_learning_openML')
