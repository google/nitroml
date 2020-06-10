import re

import requests


def get(url):
  """Sends a GET request to the given `url`."""
  resp = requests.get(url)
  if resp.status_code != requests.codes.ok:
    raise resp.raise_for_status()
  return resp


def convert_to_valid_identifier(s):
  """
  Converts the string `s` to a valid python identifier
  Remove invalid characters, and remove leading characters until letter/underscore.
  """
  s = re.sub('[^0-9a-zA-Z_]', '', s)
  s = re.sub('^[^a-zA-Z_]+', '', s)
  return s


def rename_columns(columns):
  """
  Returns a dict with keys as column and value as the new column name which is a valid python identifier.
  """
  return {column: convert_to_valid_identifier(column) for column in columns}


def parse_dataset_filters(filters):
  parsed = {}
  for filter_ in filters:
    key, value = filter_.split('=')
    parsed[key] = value
  return parsed
