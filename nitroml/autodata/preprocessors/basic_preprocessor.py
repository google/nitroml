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
"""An basic implementation of an automatic data preprocessor."""

from typing import Any, Dict, Optional, Union

from nitroml.autodata.preprocessors import preprocessor
import tensorflow as tf
import tensorflow_transform as tft

from google.protobuf import text_format
from nitroml.protos import problem_statement_pb2 as ps_pb2

Tensor = Union[tf.Tensor, tf.SparseTensor]


class BasicPreprocessor(preprocessor.Preprocessor):
  """A basic implementation of a preprocessor.

  Scales numeric features with Z-score, and computes vocabularies for
  categorical features.
  """

  PROBLEM_STATEMENT_KEY = 'problem_statement'

  @property
  def requires_inferred_feature_shapes(self) -> bool:
    """Returns whether SchemaGen should attempt to infer feature shapes."""

    return True

  @property
  def preprocessing_fn(self) -> str:
    """Returns the path to a TensorFlow Transform preprocessing_fn."""

    return 'nitroml.autodata.preprocessors.basic_preprocessor.preprocessing_fn'

  def custom_config(
      self,
      problem_statement: ps_pb2.ProblemStatement) -> Optional[Dict[str, Any]]:
    """Returns the custom config to pass to preprocessing_fn."""

    return {
        # Pass the problem statement proto as a text proto. Required since
        # custom_config must be JSON-serializable.
        self.PROBLEM_STATEMENT_KEY:
            text_format.MessageToString(
                message=problem_statement, as_utf8=True),
    }


def preprocessing_fn(inputs: Dict[str, Tensor],
                     custom_config=Dict[str, Any]) -> Dict[str, Tensor]:
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
    custom_config: Custom configuration dictionary for passing the task's
      ProblemStatement as a text proto, since custom_config must be
      JSON-serializable.

  Returns:
    Map from string feature key to transformed feature operations.
  """

  problem_statement = ps_pb2.ProblemStatement()
  text_format.Parse(
      text=custom_config[BasicPreprocessor.PROBLEM_STATEMENT_KEY],
      message=problem_statement)

  outputs = {}
  for key in [k for k, v in inputs.items() if v.dtype == tf.float32]:
    # TODO(weill): Handle case when an int field can actually represents numeric
    # rather than categorical values.
    task_type = problem_statement.tasks[0].type
    if task_type.HasField('one_dimensional_regression') and (
        key == task_type.one_dimensional_regression.label):
      # Skip normalizing regression tasks.
      continue

    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[key] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))

  for key in [k for k, v in inputs.items() if v.dtype != tf.float32]:
    # Build a vocabulary for this feature.
    # TODO(weill): Risk here to blow up computation needlessly.
    outputs[key] = tft.compute_and_apply_vocabulary(
        _fill_in_missing(inputs[key]), top_k=None, num_oov_buckets=1)

  return outputs


def _fill_in_missing(x: Tensor) -> Tensor:
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

  Args:
    x: A `Tensor` of rank 2. Its dense shape should have size at most 1 in
      the second dimension.

  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not tf.keras.backend.is_sparse(x):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
