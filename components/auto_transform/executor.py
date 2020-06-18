"""TFX AutoTransform executor definition."""

import os
from typing import Any, Dict, List, Mapping, Text

from absl import logging

import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx import types
from tfx.components.transform import executor, labels, messages
from tfx.components.util import value_utils
from tfx.types import artifact_utils
from tfx.utils import import_utils, io_utils

# custom config that we will pass to preprocessing_fn
_CUSTOM_CONFIG = 'custom_config'


class AutoTransformExecutor(executor.Executor):
  """We extend the Executor from the default transform executor and
  reimplement `Do` and `_GetPreprocessingFn` to allow a custom signature for `preprocessing_fn`.
  """

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """TensorFlow Transform executor entrypoint.

    This implements BaseExecutor.Do() and is invoked by orchestration systems.
    This is not inteded for manual usage or further customization. Please use
    the Transform() function which takes an input format with no artifact
    dependency.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_data: A list of type `standard_artifacts.Examples` which
          should contain two splits 'train' and 'eval'.
        - schema: A list of type `standard_artifacts.Schema` which should
          contain a single schema artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - transform_output: Output of 'tf.Transform', which includes an exported
          Tensorflow graph suitable for both training and serving;
        - transformed_examples: Materialized transformed examples, which
          includes both 'train' and 'eval' splits.
      exec_properties: A dict of execution properties, including either one of:
        - module_file: The file path to a python module file, from which the
          'preprocessing_fn' function will be loaded.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    train_data_uri = artifact_utils.get_split_uri(
        input_dict[executor.EXAMPLES_KEY], 'train')
    eval_data_uri = artifact_utils.get_split_uri(
        input_dict[executor.EXAMPLES_KEY], 'eval')
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict[executor.SCHEMA_KEY]))
    # # TODO(nikhilmehta) Consider adding statistics file (if needed)
    # train_statistics_file = io_utils.get_only_uri_in_dir(
    #     artifact_utils.get_single_uri(input_dict[STATISTICS_KEY]))

    transform_output = artifact_utils.get_single_uri(
        output_dict[executor.TRANSFORM_GRAPH_KEY])
    transformed_train_output = artifact_utils.get_split_uri(
        output_dict[executor.TRANSFORMED_EXAMPLES_KEY], 'train')
    transformed_eval_output = artifact_utils.get_split_uri(
        output_dict[executor.TRANSFORMED_EXAMPLES_KEY], 'eval')
    temp_path = os.path.join(transform_output,
                             executor._TEMP_DIR_IN_TRANSFORM_OUTPUT)
    logging.debug('Using temp path %s for tft.beam', temp_path)

    def _GetCachePath(label, params_dict):
      if label not in params_dict:
        return None
      else:
        return artifact_utils.get_single_uri(params_dict[label])

    label_inputs = {
        labels.COMPUTE_STATISTICS_LABEL:
            False,
        labels.SCHEMA_PATH_LABEL:
            schema_file,
        labels.EXAMPLES_DATA_FORMAT_LABEL:
            labels.FORMAT_TF_EXAMPLE,
        labels.ANALYZE_DATA_PATHS_LABEL:
            io_utils.all_files_pattern(train_data_uri),
        labels.ANALYZE_PATHS_FILE_FORMATS_LABEL:
            labels.FORMAT_TFRECORD,
        labels.TRANSFORM_DATA_PATHS_LABEL: [
            io_utils.all_files_pattern(train_data_uri),
            io_utils.all_files_pattern(eval_data_uri)
        ],
        labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: [
            labels.FORMAT_TFRECORD, labels.FORMAT_TFRECORD
        ],
        labels.MODULE_FILE:
            exec_properties.get('module_file', None),
        # TODO(b/149754658): switch to True once the TFXIO integration is
        # complete.
        labels.USE_TFXIO_LABEL:
            False,
        _CUSTOM_CONFIG:
            exec_properties.get('custom_config', None),
    }
    cache_input = _GetCachePath('cache_input_path', input_dict)
    if cache_input is not None:
      label_inputs[labels.CACHE_INPUT_PATH_LABEL] = cache_input

    label_outputs = {
        labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: transform_output,
        labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL: [
            os.path.join(transformed_train_output,
                         executor._DEFAULT_TRANSFORMED_EXAMPLES_PREFIX),
            os.path.join(transformed_eval_output,
                         executor._DEFAULT_TRANSFORMED_EXAMPLES_PREFIX),
        ],
        labels.TEMP_OUTPUT_LABEL: str(temp_path),
    }
    cache_output = _GetCachePath('cache_output_path', output_dict)
    if cache_output is not None:
      label_outputs[labels.CACHE_OUTPUT_PATH_LABEL] = cache_output
    status_file = 'status_file'  # Unused
    self.Transform(label_inputs, label_outputs, status_file)
    logging.debug('Cleaning up temp path %s on executor success', temp_path)
    io_utils.delete_dir(temp_path)

  def _GetPreprocessingFn(self, inputs: Mapping[Text, Any],
                          unused_outputs: Mapping[Text, Any],
                          custom_config) -> Any:
    """Returns a user defined preprocessing_fn.

    Args:
      inputs: A dictionary of labelled input values.
      unused_outputs: A dictionary of labelled output values.

    Returns:
      User defined function.
    """

    preprocessing_fn = import_utils.import_func_from_source(
        value_utils.GetSoleValue(inputs, labels.MODULE_FILE),
        'preprocessing_fn')

    # Note: Added custom_config here
    lambda_fn = lambda x: preprocessing_fn(x, custom_config)
    return lambda_fn

  # We override the following method.
  # The method is adapted from https://github.com/tensorflow/tfx/blob/r0.21.4/tfx/components/transform/executor.py.
  def Transform(self, inputs: Mapping[Text, Any], outputs: Mapping[Text, Any],
                status_file: Text) -> None:
    """Executes on request.

    This is the implementation part of transform executor. This is intended for
    using or extending the executor without artifact dependency.

    Args:
      inputs: A dictionary of labelled input values, including:
        - labels.COMPUTE_STATISTICS_LABEL: Whether compute statistics.
        - labels.SCHEMA_PATH_LABEL: Path to schema file.
        - labels.EXAMPLES_DATA_FORMAT_LABEL: Example data format.
        - labels.ANALYZE_DATA_PATHS_LABEL: Paths or path patterns to analyze
          data.
        - labels.ANALYZE_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          analyze data.
        - labels.TRANSFORM_DATA_PATHS_LABEL: Paths or path patterns to transform
          data.
        - labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          transform data.
        - labels.MODULE_FILE: Path to a Python module that contains the
          preprocessing_fn, optional.
        - labels.USE_TFXIO_LABEL: Whether use the TFXIO-based TFT APIs.
      outputs: A dictionary of labelled output values, including:
        - labels.PER_SET_STATS_OUTPUT_PATHS_LABEL: Paths to statistics output,
          optional.
        - labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: A path to
          TFTransformOutput output.
        - labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL: Paths to transform
          materialization.
        - labels.TEMP_OUTPUT_LABEL: A path to temporary directory.
        - _CUSTOM_CONFIG: A dict for user defined params.

      status_file: Where the status should be written (not yet implemented)
    """
    del status_file  # unused

    logging.debug('Inputs to executor.Transform function: {}'.format(inputs))
    logging.debug('Outputs to executor.Transform function: {}'.format(outputs))

    compute_statistics = value_utils.GetSoleValue(
        inputs, labels.COMPUTE_STATISTICS_LABEL)
    transform_output_path = value_utils.GetSoleValue(
        outputs, labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL)
    raw_examples_data_format = value_utils.GetSoleValue(
        inputs, labels.EXAMPLES_DATA_FORMAT_LABEL)
    schema = value_utils.GetSoleValue(inputs, labels.SCHEMA_PATH_LABEL)
    input_dataset_metadata = self._ReadMetadata(raw_examples_data_format,
                                                schema)
    use_tfxio = value_utils.GetSoleValue(inputs, labels.USE_TFXIO_LABEL)
    materialize_output_paths = value_utils.GetValues(
        outputs, labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL)
    per_set_stats_output_paths = value_utils.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
    analyze_data_paths = value_utils.GetValues(inputs,
                                               labels.ANALYZE_DATA_PATHS_LABEL)
    analyze_paths_file_formats = value_utils.GetValues(
        inputs, labels.ANALYZE_PATHS_FILE_FORMATS_LABEL)
    transform_data_paths = value_utils.GetValues(
        inputs, labels.TRANSFORM_DATA_PATHS_LABEL)
    transform_paths_file_formats = value_utils.GetValues(
        inputs, labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL)
    input_cache_dir = value_utils.GetSoleValue(
        inputs, labels.CACHE_INPUT_PATH_LABEL, strict=False)

    # CUSTOM_CONFIG
    custom_config = value_utils.GetSoleValue(
        inputs, _CUSTOM_CONFIG, strict=False)
    preprocessing_fn = self._GetPreprocessingFn(inputs, outputs, custom_config)

    output_cache_dir = value_utils.GetSoleValue(
        outputs, labels.CACHE_OUTPUT_PATH_LABEL, strict=False)
    per_set_stats_output_paths = value_utils.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
    temp_path = value_utils.GetSoleValue(outputs, labels.TEMP_OUTPUT_LABEL)

    logging.debug('Analyze data patterns: %s',
                  list(enumerate(analyze_data_paths)))
    logging.debug('Transform data patterns: %s',
                  list(enumerate(transform_data_paths)))
    logging.debug('Transform materialization output paths: %s',
                  list(enumerate(materialize_output_paths)))
    logging.debug('Transform output path: %s', transform_output_path)

    if len(analyze_data_paths) != len(analyze_paths_file_formats):
      raise ValueError(
          'size of analyze_data_paths and '
          'analyze_paths_file_formats do not match: {} v.s {}'.format(
              len(analyze_data_paths), len(analyze_paths_file_formats)))
    if len(transform_data_paths) != len(transform_paths_file_formats):
      raise ValueError(
          'size of transform_data_paths and '
          'transform_paths_file_formats do not match: {} v.s {}'.format(
              len(transform_data_paths), len(transform_paths_file_formats)))

    can_process_analysis_jointly = not bool(output_cache_dir)
    analyze_data_list = self._MakeDatasetList(analyze_data_paths,
                                              analyze_paths_file_formats,
                                              raw_examples_data_format,
                                              can_process_analysis_jointly)
    if not analyze_data_list:
      raise ValueError('Analyze data list must not be empty.')

    can_process_transform_jointly = not bool(per_set_stats_output_paths or
                                             materialize_output_paths)
    transform_data_list = self._MakeDatasetList(transform_data_paths,
                                                transform_paths_file_formats,
                                                raw_examples_data_format,
                                                can_process_transform_jointly,
                                                per_set_stats_output_paths,
                                                materialize_output_paths)

    if use_tfxio:
      all_datasets = analyze_data_list + transform_data_list
      for d in all_datasets:
        d.tfxio = self._CreateTFXIO(d, input_dataset_metadata.schema)
      self._AssertSameTFXIOSchema(all_datasets)
      feature_spec_or_typespecs = (
          all_datasets[0].tfxio.TensorAdapter().OriginalTypeSpecs())
    else:
      feature_spec_or_typespecs = schema_utils.schema_as_feature_spec(
          executor._GetSchemaProto(input_dataset_metadata)).feature_spec

      # NOTE: We disallow an empty schema, which we detect by testing the
      # number of columns.  While in principal an empty schema is valid, in
      # practice this is a sign of a user error, and this is a convenient
      # place to catch that error.
      if (not feature_spec_or_typespecs and
          not self._ShouldDecodeAsRawExample(raw_examples_data_format)):
        raise ValueError(messages.SCHEMA_EMPTY)

    # Inspecting the preprocessing_fn even if we know we need a full pass in
    # order to fail faster if it fails.
    analyze_input_columns = tft.get_analyze_input_columns(
        preprocessing_fn, feature_spec_or_typespecs)

    if not compute_statistics and not materialize_output_paths:
      if analyze_input_columns:
        logging.warning(
            'Not using the in-place Transform because the following features '
            'require analyzing: {}'.format(
                tuple(c for c in analyze_input_columns)))
      else:
        logging.warning(
            'Using the in-place Transform since compute_statistics=False, '
            'it does not materialize transformed data, and the configured '
            'preprocessing_fn appears to not require analyzing the data.')
        self._RunInPlaceImpl(preprocessing_fn, input_dataset_metadata,
                             feature_spec_or_typespecs, transform_output_path)
        # TODO(b/122478841): Writes status to status file.
        return

    materialization_format = (
        transform_paths_file_formats[-1] if materialize_output_paths else None)
    self._RunBeamImpl(use_tfxio, analyze_data_list, transform_data_list,
                      preprocessing_fn, input_dataset_metadata,
                      transform_output_path, raw_examples_data_format,
                      temp_path, input_cache_dir, output_cache_dir,
                      compute_statistics, per_set_stats_output_paths,
                      materialization_format)
