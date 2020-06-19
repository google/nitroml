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
"""An automatic transform module for the TFX Transform component."""

from typing import Dict, Text, Union

import tensorflow as tf
import tensorflow_transform as tft

Tensor = Union[tf.Tensor, tf.SparseTensor]


def preprocessing_fn(inputs: Dict[Text, Tensor]) -> Dict[Text, Tensor]:
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  for key in [k for k, v in inputs.items() if v.dtype == tf.float32]:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[key] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))

  for key in [k for k, v in inputs.items() if v.dtype != tf.float32]:
    # Build a vocabulary for this feature.
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
