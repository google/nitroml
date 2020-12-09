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
"""NitroML benchmark pipeline result overview."""

import ast
import datetime
import json
import os
import re
from typing import Dict, Any, List, NamedTuple, Optional

from nitroml.benchmark import result as br
import pandas as pd
import tensorflow.compat.v2 as tf

from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2

# Column name constants
RUN_ID_KEY = 'run_id'
STARTED_AT = 'started_at'
BENCHMARK_FULL_KEY = 'benchmark_fullname'
ARTIFACT_ID_KEY = 'artifact_id'

# Component constants
_TRAINER = 'google3.learning.elated_zebra.my_orchestrator.components.trainer.component.EstimatorTrainer'
_TRAINER_PREFIX = 'EstimatorTrainer'
_EVALUATOR = 'tfx.components.evaluator.component.Evaluator'
_KAGGLE_RESULT = 'NitroML.KaggleSubmissionResult'
_KAGGLE_PUBLISHER = 'nitroml.google.autokaggle.components.publisher.component.KagglePublisher'
_KAGGLE_PUBLISHER_PREFIX = 'KagglePublisher'
_STATS = 'ExampleStatistics'

# Name constants
_HPARAMS = 'hparams'
_NAME = 'name'
_PRODUCER_COMPONENT = 'producer_component'
_STATE = 'state'
_PIPELINE_NAME = 'pipeline_name'
_PIPELINE_ROOT = 'pipeline_root'
_RUN_ID = 'run_id'
_COMPONENT_ID = 'component_id'
_KAGGLE = 'kaggle'

# IR-Based TFXDagRunner constants
_IS_IR_KEY = 'is_ir'

# Default columns
_DEFAULT_COLUMNS = (STARTED_AT, RUN_ID_KEY,
                    br.BenchmarkResult.BENCHMARK_NAME_KEY,
                    br.BenchmarkResult.BENCHMARK_RUN_KEY,
                    br.BenchmarkResult.RUNS_PER_BENCHMARK_KEY)
_DATAFRAME_CONTEXTUAL_COLUMNS = (STARTED_AT, RUN_ID_KEY, BENCHMARK_FULL_KEY,
                                 br.BenchmarkResult.BENCHMARK_NAME_KEY,
                                 br.BenchmarkResult.BENCHMARK_RUN_KEY,
                                 br.BenchmarkResult.RUNS_PER_BENCHMARK_KEY)
_DEFAULT_CUSTOM_PROPERTIES = {
    _NAME, _PRODUCER_COMPONENT, _STATE, _PIPELINE_NAME
}


class _Result(NamedTuple):
  """Wrapper for properties and property names."""
  properties: Dict[str, Dict[str, Any]]
  property_names: List[str]


class _RunInfo(NamedTuple):
  """Wrapper for run id and component name."""
  run_id: str = ''
  component_name: str = ''
  started_at: int = 0


def _merge_results(results: List[_Result]) -> _Result:
  """Merges _Result objects into one."""
  properties = {}
  property_names = []
  for result in results:
    for key, props in result.properties.items():
      if key in properties:
        properties[key].update(props)
      else:
        properties[key] = props
    property_names += result.property_names
  return _Result(properties=properties, property_names=property_names)


def _to_pytype(val: str) -> Any:
  """Coverts val to python type."""
  try:
    return json.loads(val.lower())
  except ValueError:
    return val


def _parse_value(value: metadata_store_pb2.Value) -> Any:
  """Parse value from `metadata_store_pb2.Value` proto."""
  if value.HasField('int_value'):
    return value.int_value
  elif value.HasField('double_value'):
    return value.double_value
  else:
    return _to_pytype(value.string_value)


def _parse_hparams(hp_prop: str) -> Dict[str, Any]:
  """Parses the hparam properties string into hparams dictionary.

  Args:
    hp_prop: hparams properties retrieved from the Executor MLMD component. It
      is a serialized representation of the hparams, e.g. "['batch_size=256']"

  Returns:
    A dictionary containing hparams name and value.
  """
  # Deserialize the hparams. Execution properties are currently serialized in
  # TFX using __str__. See for details:
  # http://google3/third_party/py/tfx/orchestration/metadata.py?q=function:_update_execution_proto
  # TODO(b/151084437): Move deserialization code to TFX.
  hp_strings = ast.literal_eval(hp_prop)
  hparams = {}
  for hp in hp_strings:
    name, val = hp.split('=')
    hparams[name] = _to_pytype(val)
  return hparams


def _get_hparams(store: metadata_store.MetadataStore) -> _Result:
  """Returns the hparams of the EstimatorTrainer component.

  Args:
    store: MetaDataStore object to connect to MLMD instance.

  Returns:
    A _Result objects with properties containing hparams.
  """
  results = {}
  hparam_names = set()

  trainer_execs = store.get_executions_by_type(_TRAINER)
  for ex in trainer_execs:
    run_id = ex.properties[RUN_ID_KEY].string_value
    hparams = _parse_hparams(ex.properties[_HPARAMS].string_value)
    hparam_names.update(hparams.keys())
    hparams[RUN_ID_KEY] = run_id
    trainer_id = ex.properties[_COMPONENT_ID].string_value.replace(
        _TRAINER_PREFIX, '')
    result_key = run_id + trainer_id
    hparams[br.BenchmarkResult.BENCHMARK_NAME_KEY] = trainer_id[
        1:]  # Removing '.' prefix
    # BeamDagRunner uses iso format timestamp. See for details:
    # http://google3/third_party/py/tfx/orchestration/beam/beam_dag_runner.py
    try:
      hparams[STARTED_AT] = datetime.datetime.fromtimestamp(int(run_id))
    except ValueError:
      hparams[STARTED_AT] = run_id
    results[result_key] = hparams
  return _Result(properties=results, property_names=sorted(hparam_names))


def _get_artifact_run_info_map(store: metadata_store.MetadataStore,
                               artifact_ids: List[int]) -> Dict[int, _RunInfo]:
  """Returns a dictionary mapping artifact_id to its MyOrchestrator run_id.

  Args:
    store: MetaDataStore object to connect to MLMD instance.
    artifact_ids: A list of artifact ids to load.

  Returns:
    A dictionary containing artifact_id as a key and MyOrchestrator run_id as value.
  """
  # Get events of artifacts.
  events = store.get_events_by_artifact_ids(artifact_ids)
  exec_to_artifact = {}
  for event in events:
    exec_to_artifact[event.execution_id] = event.artifact_id

  # Get execution of artifacts.
  executions = store.get_executions_by_id(list(exec_to_artifact.keys()))
  artifact_to_run_info = {}
  for execution in executions:
    run_id = execution.properties[RUN_ID_KEY].string_value
    component = execution.properties[_COMPONENT_ID].string_value
    artifact_id = exec_to_artifact[execution.id]
    artifact_to_run_info[artifact_id] = _RunInfo(
        run_id=run_id,
        component_name=component,
        started_at=execution.create_time_since_epoch)

  return artifact_to_run_info


def _get_benchmark_results(store: metadata_store.MetadataStore) -> _Result:
  """Returns the benchmark results of the BenchmarkResultPublisher component.

  Args:
    store: MetaDataStore object to connect to MLMD instance.

  Returns:
    A _Result objects with properties containing benchmark results.
  """
  metrics = {}
  property_names = set()
  publisher_artifacts = store.get_artifacts_by_type(
      br.BenchmarkResult.TYPE_NAME)
  for artifact in publisher_artifacts:
    evals = {_IS_IR_KEY: False}
    for key, val in artifact.custom_properties.items():
      evals[key] = _parse_value(val)
      # Change for the IR world.
      if key == 'name':
        new_id = _parse_value(val).split(':')
        if len(new_id) > 2:
          evals[RUN_ID_KEY] = new_id[1]
          evals[_IS_IR_KEY] = True
    property_names = property_names.union(evals.keys())
    metrics[artifact.id] = evals

  artifact_to_run_info = _get_artifact_run_info_map(store, list(metrics.keys()))

  properties = {}
  for artifact_id, evals in metrics.items():
    run_info = artifact_to_run_info[artifact_id]
    if evals[_IS_IR_KEY]:
      started_at = run_info.started_at // 1000
      evals[STARTED_AT] = datetime.datetime.fromtimestamp(started_at)
      run_id = metrics[artifact_id][RUN_ID_KEY]
    else:
      run_id = run_info.run_id
      evals[RUN_ID_KEY] = run_id
      # Old BeamDagRunner uses iso format timestamp. See for details:
      # http://google3/third_party/py/tfx/orchestration/beam/beam_dag_runner.py
      try:
        evals[STARTED_AT] = datetime.datetime.fromtimestamp(int(run_id))
      except ValueError:
        evals[STARTED_AT] = run_id
    evals.pop(_IS_IR_KEY)
    result_key = run_id + '.' + evals[br.BenchmarkResult.BENCHMARK_NAME_KEY]
    properties[result_key] = evals

  property_names = property_names.difference(
      {_NAME, _PRODUCER_COMPONENT, _STATE, *_DEFAULT_COLUMNS, _IS_IR_KEY})
  return _Result(properties=properties, property_names=sorted(property_names))


def _get_kaggle_results(store: metadata_store.MetadataStore) -> _Result:
  """Returns the kaggle score detail from the KagglePublisher component.

  Args:
    store: MetaDataStore object to connect to MLMD instance.

  Returns:
    A _Result objects with properties containing kaggle results.
  """
  results = {}
  property_names = set()
  kaggle_artifacts = store.get_artifacts_by_type(_KAGGLE_RESULT)
  for artifact in kaggle_artifacts:
    submit_info = {_IS_IR_KEY: False}
    for key, val in artifact.custom_properties.items():
      if key not in _DEFAULT_CUSTOM_PROPERTIES:
        name = _KAGGLE + '_' + key
        submit_info[name] = _parse_value(val)
      # Change for the IR world.
      if key == 'name':
        new_id = _parse_value(val).split(':')
        if len(new_id) > 2:
          submit_info[RUN_ID_KEY] = new_id[1]
          submit_info[_IS_IR_KEY] = True
    property_names = property_names.union(submit_info.keys())
    results[artifact.id] = submit_info

  artifact_to_run_info = _get_artifact_run_info_map(store, list(results.keys()))

  properties = {}
  for artifact_id, submit_info in results.items():
    run_info = artifact_to_run_info[artifact_id]
    if submit_info[_IS_IR_KEY]:
      run_id = results[artifact_id][RUN_ID_KEY]
    else:
      run_id = run_info.run_id
    result_key = run_id + run_info.component_name.replace(
        _KAGGLE_PUBLISHER_PREFIX, '')
    submit_info.pop(_IS_IR_KEY)
    properties[result_key] = submit_info

  property_names = property_names.difference(
      {_NAME, _PRODUCER_COMPONENT, _STATE, *_DEFAULT_COLUMNS, _IS_IR_KEY})
  return _Result(properties=properties, property_names=sorted(property_names))


def get_model_dir_map(store: metadata_store.MetadataStore) -> Dict[str, str]:
  """Obtains a map of run_id to model_dir from the store."""

  evaluator_execs = store.get_executions_by_type(_EVALUATOR)

  def _go_up_2_levels(eval_model_dirs):
    model_dir_set = set()
    for eval_model_dir in eval_model_dirs:
      model_dir_set.add(os.sep.join(eval_model_dir.split(os.sep)[:-2]))
    return list(model_dir_set)

  def _eval_execs_to_model_dir_map(eval_execs):
    model_dir_map = {}
    for eval_exec in eval_execs:
      run_id = eval_exec.properties[_RUN_ID].string_value
      pipeline_root = eval_exec.properties[_PIPELINE_ROOT].string_value
      eval_component_id = eval_exec.properties[_COMPONENT_ID].string_value
      eval_config_path = os.path.join(pipeline_root,
                                      eval_component_id, 'evaluation',
                                      str(eval_exec.id), 'eval_config.json')

      with tf.io.gfile.GFile(eval_config_path, 'r') as f:
        eval_config = json.load(f)

      model_dir_map[run_id] = _go_up_2_levels(
          eval_config['modelLocations'].values())
    return model_dir_map

  return _eval_execs_to_model_dir_map(evaluator_execs)


def get_statisticsgen_dir_list(
    store: metadata_store.MetadataStore) -> List[str]:
  """Obtains a list of statisticsgen_dir from the store."""

  stats_artifacts = store.get_artifacts_by_type(_STATS)
  stat_dirs_list = [artifact.uri for artifact in stats_artifacts]
  return stat_dirs_list


def _make_dataframe(metrics_list: List[Dict[str, Any]],
                    columns: List[str]) -> pd.DataFrame:
  """Makes pandas.DataFrame from metrics_list."""
  df = pd.DataFrame(metrics_list)
  if not df.empty:
    # Reorder columns.
    # Strip benchmark run repetition for aggregation.
    df[BENCHMARK_FULL_KEY] = df[br.BenchmarkResult.BENCHMARK_NAME_KEY]
    df[br.BenchmarkResult.BENCHMARK_NAME_KEY] = df[
        br.BenchmarkResult.BENCHMARK_NAME_KEY].apply(
            lambda x: re.sub(r'\.run_\d_of_\d$', '', x))

    key_columns = list(_DATAFRAME_CONTEXTUAL_COLUMNS)
    if br.BenchmarkResult.BENCHMARK_RUN_KEY not in df:
      key_columns.remove(br.BenchmarkResult.BENCHMARK_RUN_KEY)
    if br.BenchmarkResult.RUNS_PER_BENCHMARK_KEY not in df:
      key_columns.remove(br.BenchmarkResult.RUNS_PER_BENCHMARK_KEY)
    df = df[key_columns + columns]

    df = df.set_index([STARTED_AT])

  return df


def _aggregate_results(df: pd.DataFrame,
                       metric_aggregators: Optional[List[Any]],
                       groupby_columns: List[str]):
  """Aggregates metrics in an overview pd.DataFrame."""

  df = df.copy()
  groupby_columns = groupby_columns.copy()
  if br.BenchmarkResult.BENCHMARK_RUN_KEY in df:
    df = df.drop([br.BenchmarkResult.BENCHMARK_RUN_KEY], axis=1)
  groupby_columns.remove(br.BenchmarkResult.BENCHMARK_RUN_KEY)
  groupby_columns.remove(BENCHMARK_FULL_KEY)
  if br.BenchmarkResult.RUNS_PER_BENCHMARK_KEY not in df:
    groupby_columns.remove(br.BenchmarkResult.RUNS_PER_BENCHMARK_KEY)

  # Group by contextual columns and aggregate metrics.
  df = df.groupby(groupby_columns)
  df = df.agg(metric_aggregators)

  # Flatten MultiIndex into a DataFrame.
  df.columns = [' '.join(col).strip() for col in df.columns.values]
  return df.reset_index().set_index('started_at')


def overview(
    store: metadata_store.MetadataStore,
    metric_aggregators: Optional[List[Any]] = None,
) -> pd.DataFrame:
  """Returns a pandas.DataFrame containing hparams and evaluation results.

  This method assumes that `tf.enable_v2_behavior()` was called beforehand.
  It loads results for all evaluation therefore method can be slow.

  TODO(b/151085210): Allow filtering incomplete benchmark runs.

  Assumptions:
    For the given pipeline, MyOrchestrator run_id and component_id of trainer is unique
    and (my_orchestrator_run_id + trainer.component_id-postfix) is equal to
    (my_orchestrator_run_id + artifact.producer_component-postfix).

  Args:
    store: MetaDataStore object for connecting to an MLMD instance.
    metric_aggregators: Iterable of functions and/or function names, e.g.
      [np.sum, 'mean']. Groups individual runs by their contextual features (run
      id, hparams), and aggregates metrics by the given functions. If a
      function, must either work when passed a DataFrame or when passed to
      DataFrame.apply.

  Returns:
    A pandas DataFrame with the loaded hparams and evaluations or an empty one
    if no evaluations and hparams could be found.
  """
  hparams_result = _get_hparams(store)
  metrics_result = _get_benchmark_results(store)
  kaggle__result = _get_kaggle_results(store)

  # Merge results
  result = _merge_results([hparams_result, metrics_result, kaggle__result])

  # Filter metrics that have empty hparams and evaluation results.
  results_list = [
      result for result in result.properties.values()
      if len(result) > len(_DEFAULT_COLUMNS)
  ]

  df = _make_dataframe(results_list, result.property_names)
  if metric_aggregators:
    return _aggregate_results(
        df,
        metric_aggregators=metric_aggregators,
        groupby_columns=list(_DATAFRAME_CONTEXTUAL_COLUMNS) +
        hparams_result.property_names)
  return df
