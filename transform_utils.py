"""
The module file for the Transform component.
"""
from typing import Any, Dict, Text

import tensorflow as tf
import tensorflow_transform as tft
from absl import logging


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def preprocessing_fn(inputs: Dict[Text, Any],
                     custom_config: Dict[Text, Any]) -> Dict[Text, Any]:
  """tf.transform's callback fn for preprocessing inputs with custom signature.

  Args:
    inputs: Map from string feature key to raw not-yet-transformed features.

    Note: Transformations within the preprocessing_fn cannot be applied to the label feature for training or eval.
    (See: https://www.tensorflow.org/tfx/guide/keras)

  Returns:
    outputs: Map from string feature key to transformed feature operations.
  """
  outputs = inputs.copy()
  logging.info(custom_config)

  if not custom_config:
    raise ValueError('Parameter `custom_config` required.')

  label_key = custom_config['label_key']

  for key in outputs:
    outputs[key] = _fill_in_missing(outputs[key])

  labels = outputs[label_key]
  tft.vocabulary(labels, vocab_filename='label_vocab')
  return outputs
