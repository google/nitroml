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
"""Utility module for suites."""

import re
from typing import Dict, List

import requests


def get(url: str, **kwargs) -> requests.Response:
  """Sends a GET request to the given `url`."""

  resp = requests.get(url, **kwargs)
  resp.raise_for_status()
  return resp


def convert_to_valid_identifier(s: str) -> str:
  """Converts the string `s` to a valid python identifier.

  Removes invalid characters, and any leading characters until
  letter/underscore.

  Args:
    s: input string.

  Returns:
    `si` with invalid characters removed, and any leading characters until
    letter/underscore.
  """

  s = re.sub('[^0-9a-zA-Z_]', '', s)
  s = re.sub('^[^a-zA-Z_]+', '', s)
  return s


def rename_columns(columns: List[str]) -> Dict[str, str]:
  """Returns dict[col_name] = new_col_name."""

  return {column: convert_to_valid_identifier(column) for column in columns}


def parse_dataset_filters(filters: List[str]) -> Dict[str, str]:
  """Coverts string array of the form ["key=value"] to dict[key]=value."""

  return dict(tuple(f.split('=')) for f in filters)
