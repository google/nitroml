import re
from typing import Dict, List, Text

import requests


def get(url: Text) -> Text:
  """Sends a GET request to the given `url`."""
  resp = requests.get(url)
  if resp.status_code != requests.codes.ok:
    raise resp.raise_for_status()
  return resp


def convert_to_valid_identifier(s: Text) -> Text:
  """Converts the string `s` to a valid python identifier
  Remove invalid characters, and remove leading characters until letter/underscore.
  """
  s = re.sub('[^0-9a-zA-Z_]', '', s)
  s = re.sub('^[^a-zA-Z_]+', '', s)
  return s


def rename_columns(columns: Dict[Text, Text]) -> Dict[Text, Text]:
  """Returns a dict with keys as column and value as the new column name which is a valid python identifier."""
  return {column: convert_to_valid_identifier(column) for column in columns}


def parse_dataset_filters(filters: List[Text]) -> Dict[Text, Text]:
  parsed = {}
  for filter_ in filters:
    key, value = filter_.split('=')
    parsed[key] = value
  return parsed
