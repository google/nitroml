"""Serialization methods for context properties."""

import json
from typing import Dict, Union

ContextDict = Dict[str, Union[int, float, str, bool]]


def _type_check(properties):
  for key, value in properties.items():
    if not isinstance(key, str):
      raise TypeError('Key (%s}) is of type (%s), str required' %
                      (str(key), type(key)))
    if not isinstance(value, (int, float, str, bool)):
      raise TypeError(
          'Value (%s}) is of type (%s), int, float, or str required' %
          (str(value), type(value)))


def decode(properties: str) -> ContextDict:
  """Converts a serialized property dict into a python dict.

  Args:
    properties: A string rep of a properties dictionary encoded by encode().
  Returns:
    Dict representation of properties.
  """
  return json.loads(properties)


def encode(properties: ContextDict) -> str:
  """Convert a property dict to a serializeabled string.

  Args:
    properties: A dictionary of string keys and primitive datatype values.
  Returns:
    A serialized json string rep of properties.

  Raises:
    TypeError: If any of the values in the dictionary are not serializeable.
    Current support for int, float, and string datatypes
  """
  _type_check(properties)
  return json.dumps(properties, sort_keys=True)

