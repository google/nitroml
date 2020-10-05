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
from typing import Any, Mapping, Text

from tfx import components
from tfx.components.transform import executor
from tfx.components.transform import labels


class Executor(executor.Executor):
  """Transform executor."""

  def _GetPreprocessingFn(self, inputs: Mapping[Text, Any],
                          outputs: Mapping[Text, Any]) -> Any:
    """See base class."""
    fn = super(Executor, self)._GetPreprocessingFn(inputs, outputs)
    # The internal function's inputs argument shadows the external inputs
    # argument. We can't rename either in case inputs are specified as kwargs,
    # so we keep a reference to the external inputs here instead.
    inputs_dict = inputs

    def preprocessing_fn(inputs):
      """The preprocessing_fn to return."""
      kwargs = {}
      argspec = inspect.getfullargspec(fn).args

      if 'schema' in argspec:
        schema_path = components.util.value_utils.GetSoleValue(
            inputs_dict, labels.SCHEMA_PATH_LABEL)
        schema = self._GetSchema(schema_path)
        kwargs['schema'] = schema

      if 'transform_graph_dir' in argspec:
        kwargs[
            'transform_graph_dir'] = components.util.value_utils.GetSoleValue(
                outputs, labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL)
      return fn(inputs, **kwargs)

    return preprocessing_fn
