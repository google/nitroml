import datetime
import os
import sys

_HOME = os.environ['HOME']
sys.path.append(os.path.join(_HOME, 'airflow', 'dags'))
sys.path.append(os.path.join(_HOME, 'airflow', 'dags', 'datasets'))

from typing import Text

import tensorflow_model_analysis as tfma

from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import Evaluator
from tfx.components import Pusher

from tfx.components.trainer.executor import GenericExecutor

from tfx.orchestration import pipeline
from tfx.orchestration import metadata

from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig

from datasets.dataset import OpenMLDataset
from tfx.proto import trainer_pb2
from tfx.components.base import executor_spec

_pipeline_name = 'starter_project_pipeline'

_workspace_root = os.path.join(_HOME, 'nitroml')
_pipeline_root = os.path.join(_HOME, 'pipeline')
_metadata_path = os.path.join(_pipeline_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Specify the dataset path
_data_path = os.path.join(_HOME, 'output')

_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2020, 6, 9)
}

# These should be generic modules that can work with
_transform_module_file = os.path.join(_workspace_root, 'transform_utils.py')
_trainer_module_file = os.path.join(_workspace_root, 'trainer_utils.py')


def get_evaluation_config(label_key: Text = ''):

  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key=label_key)],
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

  return eval_config


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, datasets: object,
                     transform_module_file: Text, trainer_module_file: Text,
                     metadata_path: Text):

  all_example_gens = datasets.components
  all_tasks = datasets.tasks

  assert len(all_tasks) == len(all_example_gens)

  print(f'A Total of {len(all_example_gens)} benchmarks.')

  example_gen = all_example_gens[0]
  task = all_tasks[0]

  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)

  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=transform_module_file)

  trainer_config = task.toJSON()
  print(trainer_config)
  # trainer_config = {'task': json}
  trainer = Trainer(
      module_file=trainer_module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      custom_config=trainer_config,
      train_args=trainer_pb2.TrainArgs(num_steps=10),
      eval_args=trainer_pb2.EvalArgs(num_steps=5))

  eval_config = get_evaluation_config()

  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config)

  num_workers = 2
  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          transform,
          trainer,
          evaluator,
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=[f'--direct_num_workers={num_workers}'])


datasets = OpenMLDataset(_data_path)
pipeline_config = AirflowPipelineConfig(_airflow_config)
DAG = AirflowDagRunner(pipeline_config)
pipeline = _create_pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    datasets=datasets,
    transform_module_file=_transform_module_file,
    trainer_module_file=_trainer_module_file,
    metadata_path=_metadata_path)

DAG = DAG.run(pipeline)
