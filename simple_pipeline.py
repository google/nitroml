"""
This script prepares the tfx pipeline that can run with Airflow.
"""
import datetime
import os
import sys
from typing import List

import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from absl import logging
from tfx.components import Evaluator, SchemaGen, StatisticsGen, Trainer
from tfx.components.base import base_component, executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.airflow.airflow_dag_runner import (
    AirflowDagRunner, AirflowPipelineConfig)
from tfx.proto import trainer_pb2

from components import AutoTransform
from datasets.dataset import OpenMLDataset

_HOME = os.environ['HOME']
_PIPELINE_NAME = 'all_benchmarks'
_WORKSPACE_ROOT = os.path.join(_HOME, 'nitroml')

_TRANSFORM_MODULE_FILE = os.path.join(_WORKSPACE_ROOT, 'transform_utils.py')
_TRAINER_MODULE_FILE = os.path.join(_WORKSPACE_ROOT, 'trainer_utils.py')
_PIPELINE_ROOT = os.path.join(_HOME, 'pipeline')
_METADATA_PATH = os.path.join(_PIPELINE_ROOT, 'metadata', _PIPELINE_NAME,
                              'metadata.db')
_DATA_PATH = os.path.join(_HOME, 'output')
_AIRFLOW_CONFIG = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2020, 6, 9)
}


def _create_pipeline(components: List[base_component.BaseComponent]
                    ) -> pipeline.Pipeline:

  num_workers = 2
  return pipeline.Pipeline(
      pipeline_name=_PIPELINE_NAME,
      pipeline_root=_PIPELINE_ROOT,
      components=components,
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          _METADATA_PATH),
      beam_pipeline_args=[f'--direct_num_workers={num_workers}'])


def _create_benchmarks(datasets: object):

  all_example_gens = datasets.components
  all_tasks = datasets.tasks
  all_names = datasets.names

  assert len(all_tasks) == len(all_example_gens)

  all_example_gens = all_example_gens[:3]
  all_tasks = all_tasks[:3]
  all_names = all_names[:3]
  # all_example_gens = all_example_gens[2:3]
  # all_tasks = all_tasks[2:3]
  # all_names = all_names[2:3]
  # index = all_names.index('car')
  # all_example_gens = all_example_gens[index:index + 1]
  # all_tasks = all_tasks[index:index + 1]
  # all_names = all_names[index:index + 1]

  logging.info(f'A Total of {len(all_example_gens)} benchmarks.')

  all_components = []

  for example_gen, task, dataset_name in zip(all_example_gens, all_tasks,
                                             all_names):

    stats_options = tfdv.StatsOptions(label_feature=task.label_key)
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'],
        stats_options=stats_options,
        instance_name=dataset_name)

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False,
        instance_name=dataset_name)

    transform_config = task.toJSON()
    transform = AutoTransform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=_TRANSFORM_MODULE_FILE,
        instance_name=dataset_name,
        custom_config=transform_config)

    trainer_config = task.toJSON()
    trainer = Trainer(
        module_file=_TRAINER_MODULE_FILE,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        custom_config=trainer_config,
        train_args=trainer_pb2.TrainArgs(num_steps=1),
        eval_args=trainer_pb2.EvalArgs(num_steps=1),
        instance_name=dataset_name)

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=task.label_key)],
        metrics_specs=[
            tfma.MetricsSpec(
                # The metrics added here are in addition to those saved with the
                # model (assuming either a keras model or EvalSavedModel is used).
                # Any metrics added into the saved model (for example using
                # model.compile(..., metrics=[...]), etc) will be computed
                # automatically.
                # To add validation thresholds for metrics saved with the model,
                # add them keyed by metric name to the thresholds map.
                metrics=[
                    tfma.MetricConfig(class_name='ExampleCount'),
                    # tfma.MetricConfig(
                    #     class_name='CategoricalAccuracy',
                    #     threshold=tfma.MetricThreshold(
                    #         value_threshold=tfma.GenericValueThreshold(
                    #             lower_bound={'value': 0.1}),
                    #         change_threshold=tfma.GenericChangeThreshold(
                    #             direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                    #             absolute={'value': -1e-10})))
                ])
        ],
        slicing_specs=[tfma.SlicingSpec()])

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config,
        instance_name=dataset_name)

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        transform,
        trainer,
        evaluator,
    ]

    all_components += components

  return all_components


all_datasets = OpenMLDataset(_DATA_PATH)
pipeline_config = AirflowPipelineConfig(_AIRFLOW_CONFIG)
DAG = AirflowDagRunner(pipeline_config)

components = _create_benchmarks(datasets=all_datasets)

logging.info(f'There will be {len(components)} total components in the DAG.')
pipeline = _create_pipeline(components)

DAG = DAG.run(pipeline)
