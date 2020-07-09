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
"""Executor for Transform."""

import inspect
from typing import Any, Dict, List, Mapping, Text

from tfx import components
from tfx import types
from tfx.components.transform import executor
from tfx.components.transform import labels


class Executor(executor.Executor):
  """Transform executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """See base class."""

    # TODO(b/148932926): This should be handled in core TFX Transform.
    self._custom_config = exec_properties['custom_config']
    super(Executor, self).Do(input_dict, output_dict, exec_properties)

  def _GetPreprocessingFn(self, inputs: Mapping[Text, Any],
                          unused_outputs: Mapping[Text, Any]) -> Any:
    """See base class."""
    fn = super(Executor, self)._GetPreprocessingFn(inputs, unused_outputs)
    # The internal function's inputs argument shadows the external inputs
    # argument. We can't rename either in case inputs are specified as kwargs,
    # so we keep a reference to the external inputs here instead.
    inputs_dict = inputs

    def preprocessing_fn(inputs):
      """The preprocessing_fn to return."""
      args = {}
      argspec = inspect.getfullargspec(fn).args
      if 'schema' in argspec:
        schema_path = components.util.value_utils.GetSoleValue(
            inputs_dict, labels.SCHEMA_PATH_LABEL)
        schema = self._GetSchema(schema_path)
        args['schema'] = schema
      if 'custom_config' in argspec:
        args['custom_config'] = self._custom_config
      return fn(inputs, **args)

    return preprocessing_fn
