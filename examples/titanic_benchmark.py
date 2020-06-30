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

from absl import logging
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from nitroml.datasets import tfds_dataset
from examples import config
from nitroml import nitroml
import tensorflow_datasets as tfds

from tfx import components as tfx
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.proto import trainer_pb2
# pylint: enable=g-import-not-at-top

USE_KUBEFLOW = True

class TitanicBenchmark(nitroml.Benchmark):
  r"""Demos a NitroML benchmark on the 'Titanic' dataset from OpenML."""

  def benchmark(self, **kwargs):
    # NOTE: For convenience, we fetch the OpenML task from the AutoTFX
    # tasks repository.
    data_dir = kwargs.get('data_dir', None)
    dataset = tfds_dataset.TFDSDataset(tfds.builder('titanic', data_dir=data_dir))

    # Compute dataset statistics.
    statistics_gen = tfx.StatisticsGen(examples=dataset.examples)

    # Infer the dataset schema.
    schema_gen = tfx.SchemaGen(
        statistics=statistics_gen.outputs.statistics, infer_feature_shape=True)

    # Apply global transformations and compute vocabularies.
    transform = tfx.Transform(
        examples=dataset.examples,
        schema=schema_gen.outputs.schema,
        preprocessing_fn='examples.auto_transform.preprocessing_fn')

    # Define a tf.estimator.Estimator-based trainer.
    trainer = tfx.Trainer(
        run_fn='examples.auto_estimator_trainer.run_fn',
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor),
        transformed_examples=transform.outputs.transformed_examples,
        schema=schema_gen.outputs.schema,
        transform_graph=transform.outputs.transform_graph,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))

    # Collect the pipeline components to benchmark.
    pipeline = dataset.components + [
        statistics_gen, schema_gen, transform, trainer
    ]

    # Finally, call evaluate() on the workflow DAG outputs. This will
    # automatically append Evaluators to compute metrics from the given
    # SavedModel and 'eval' TF Examples.
    self.evaluate(
        pipeline, examples=dataset.examples, model=trainer.outputs.model)

def get_kubeflow_dag_runner():
  
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)
  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      tfx_image=tfx_image
  )

  return kubeflow_dag_runner.KubeflowDagRunner(config=runner_config)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
    
  if USE_KUBEFLOW:
    pipeline_root = os.path.join('gs://', config.GCS_BUCKET_NAME, config.PIPELINE_NAME)
    download_dir = os.path.join('gs://', config.GCS_BUCKET_NAME, 'tensorflow-datasets')
    tfx_runner = get_kubeflow_dag_runner()
    nitroml.main(pipeline_name=config.PIPELINE_NAME, pipeline_root=pipeline_root, data_dir=download_dir, tfx_runner=tfx_runner)
  else:
    nitroml.main()
