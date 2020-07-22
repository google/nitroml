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
r"""Meta Learning helper class the defines the meta learning DAG."""

from typing import Any, Dict, Optional, Text, Union


class MetaLearningWrapper(object):
  """A helper class that wraps definition of meta learning sub-pipeline."""

  def __init__(self,
               train_transformed_examples: List,
               train_stats_gens: List,
               test_transformed_examples: List,
               test_stats_gens: List,
               algorithm: Text = 'nearest_neighbor'):

    self._train_transformed_examples = train_transformed_examples
    self._train_stats_gens = train_stats_gens
    self._test_transformed_examples = test_transformed_examples
    self._test_stats_gens = test_stats_gens
    self._algorithm = algorithm

  def _build_pipeline(self):
    """Builds the meta-learning pipeline."""

    for stats_gen in self._train_statistics:
      