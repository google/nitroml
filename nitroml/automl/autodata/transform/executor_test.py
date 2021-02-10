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
"""Tests for nitroml.automl.autodata.transform.executor."""

import json
import os
import tempfile

from absl import flags
from absl.testing import absltest
from nitroml.automl.autodata.transform import executor as my_orchestrator_executor
import tensorflow.compat.v2 as tf
import tensorflow_transform as tft
from tfx import types
from tfx.components.transform import executor as tfx_executor
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class _TempPath(types.Artifact):
  TYPE_NAME = 'TempPath'


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    # Create input_dict.
    self._input_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(self._input_data_dir, 'example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    schema_artifact = standard_artifacts.Schema()
    schema_artifact.uri = os.path.join(self._input_data_dir, 'schema_gen')
    self._input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
        standard_component_specs.SCHEMA_KEY: [schema_artifact],
    }

    # Create output_dict.
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                       tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)),
        self._testMethodName)
    self._transformed_output = standard_artifacts.TransformGraph()
    self._transformed_output.uri = os.path.join(output_data_dir,
                                                'transformed_output')
    self._transformed_examples = standard_artifacts.Examples()
    self._transformed_examples.uri = output_data_dir
    self._transformed_examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    temp_path_output = _TempPath()
    temp_path_output.uri = tempfile.mkdtemp()
    self._output_dict = {
        standard_component_specs.TRANSFORM_GRAPH_KEY: [
            self._transformed_output
        ],
        standard_component_specs.TRANSFORMED_EXAMPLES_KEY: [
            self._transformed_examples
        ],
        tfx_executor.TEMP_PATH_KEY: [temp_path_output],
    }

    # Create exec properties.
    self._exec_properties = {
        'custom_config':
            json.dumps({'problem_statement_path': '/some/fake/path'})
    }

  def _verify_transform_outputs(self):
    self.assertNotEmpty(
        tf.io.gfile.listdir(
            os.path.join(self._transformed_examples.uri, 'train')))
    self.assertNotEmpty(
        tf.io.gfile.listdir(
            os.path.join(self._transformed_examples.uri, 'eval')))
    path_to_saved_model = os.path.join(self._transformed_output.uri,
                                       tft.TFTransformOutput.TRANSFORM_FN_DIR,
                                       tf.saved_model.SAVED_MODEL_FILENAME_PB)
    self.assertTrue(tf.io.gfile.exists(path_to_saved_model))

  def test_do_with_preprocessing_fn_vanilla(self):
    preprocessing_fn = ('nitroml.automl.autodata.transform.testdata.'
                        'preprocessing_fns.preprocessing_fn_vanilla')
    self._exec_properties['preprocessing_fn'] = preprocessing_fn
    executor = my_orchestrator_executor.Executor()

    executor.Do(self._input_dict, self._output_dict, self._exec_properties)

    self._verify_transform_outputs()

  def test_do_with_preprocessing_fn_richer(self):
    preprocessing_fn = ('nitroml.automl.autodata.transform.testdata.'
                        'preprocessing_fns.preprocessing_fn_richer')
    self._exec_properties['preprocessing_fn'] = preprocessing_fn
    executor = my_orchestrator_executor.Executor()

    executor.Do(self._input_dict, self._output_dict, self._exec_properties)

    self._verify_transform_outputs()


if __name__ == '__main__':
  absltest.main()
