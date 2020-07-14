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
# Required since Python binaries ignore relative paths when importing:
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import nitroml
from nitroml.components.transform import component
from nitroml.datasets import openml_cc18
from examples import config
from examples import auto_keras_trainer
from tfx import components as tfx
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.proto import trainer_pb2


class OpenMLCC18Benchmark(nitroml.Benchmark):
  r"""Demos a NitroML benchmark on the 'OpenML-CC18' classification datasets."""

  def benchmark(self,
                mock_data: bool = False,
                data_dir: str = None,
                use_keras: bool = True):

    # TODO(nikhilmehta): create subbenchmarks using all 72 datasets
    datasets = openml_cc18.OpenMLCC18(data_dir, mock_data=mock_data)

    if mock_data:
      dataset_indices = [0]
    else:
      dataset_indices = range(20, 40)

    # List of datasets that do not incur OOM - [4,11]
    for ix in dataset_indices:
      name = datasets.names[ix]
      with self.sub_benchmark(name):
        example_gen = datasets.components[ix]
        task = datasets.tasks[ix]
        task_dict = task.to_dict()
        task_dict.pop('description')

        statistics_gen = tfx.StatisticsGen(
            examples=example_gen.outputs.examples)

        schema_gen = tfx.SchemaGen(
            statistics=statistics_gen.outputs.statistics,
            infer_feature_shape=True)

        transform = component.Transform(
            examples=example_gen.outputs.examples,
            schema=schema_gen.outputs.schema,
            preprocessing_fn='examples.auto_transform.preprocessing_fn')

        run_fn = 'examples.auto_keras_trainer.run_fn' if use_keras else 'examples.auto_estimator_trainer.run_fn'
        trainer = tfx.Trainer(
            run_fn=run_fn,
            custom_executor_spec=executor_spec.ExecutorClassSpec(
                trainer_executor.GenericExecutor),
            transformed_examples=transform.outputs.transformed_examples,
            schema=schema_gen.outputs.schema,
            transform_graph=transform.outputs.transform_graph,
            train_args=trainer_pb2.TrainArgs(num_steps=1),
            eval_args=trainer_pb2.EvalArgs(num_steps=1),
            custom_config=task_dict)

        # Collect the pipeline components to benchmark.
        pipeline = [example_gen, statistics_gen, schema_gen, transform, trainer]

        eval_config = auto_keras_trainer.get_eval_config(
            task) if use_keras else None
        # Finally, call evaluate() on the workflow DAG outputs, This will
        # automatically append Evaluators to compute metrics from the given
        # SavedModel and 'eval' TF Examples.
        self.evaluate(
            pipeline,
            examples=example_gen.outputs['examples'],
            model=trainer.outputs.model,
            eval_config=eval_config)


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
    nitroml.main()
