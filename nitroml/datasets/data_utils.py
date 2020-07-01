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
"""Utility module for datasets"""

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
  """Returns a dict with keys as column_name and value as the new column_name
  which is a valid python identifier.
  """

  return {column: convert_to_valid_identifier(column) for column in columns}


def parse_dataset_filters(filters: List[Text]) -> Dict[Text, Text]:
  """Parse the dataset filters: Coverts string array of form ["key=value"] to dict[key]=value.
  """

  parsed = {}
  for filter_ in filters:
    key, value = filter_.split('=')
    parsed[key] = value
  return parsed
