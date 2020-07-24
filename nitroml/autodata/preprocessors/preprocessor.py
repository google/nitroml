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
"""Automated data preprocessors."""

import abc
from typing import Any, Dict, Optional

from nitroml.protos import problem_statement_pb2 as ps_pb2


class Preprocessor(abc.ABC):
  """Defines the automated preprocessing to apply to raw data.

  Subclasses are responsible for defining automated feature-preprocessing,
  feature-engineering, and feature-selection for downstream Trainers and Tuners.

  Preprocessors specify ops for TensorFlow Transform to preprocess data for
  AutoML researchers who want to focus on downstream AutoML functions such as
  automated model-search and automated ensembling.
  """

  @abc.abstractproperty
  def requires_inferred_feature_shapes(self) -> bool:
    """Returns whether SchemaGen should attempt to infer feature shapes."""

  @abc.abstractproperty
  def preprocessing_fn(self) -> str:
    """Returns the path to a TensorFlow Transform preprocessing_fn."""

  @abc.abstractmethod
  def custom_config(
      self,
      problem_statement: ps_pb2.ProblemStatement) -> Optional[Dict[str, Any]]:
    """Returns the custom config to pass to preprocessing_fn."""
