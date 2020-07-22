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
"""Executor for MetaFeatureGen."""


class MetaFeatureGenExecutor(base_executor.BaseExecutor):
  """Executor for MetaFeatureGen."""

  def Do(self, input_dict: Dict[Text, List[Artifact]],
         output_dict: Dict[Text, List[Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Generate MetaFeatures for meta training datasets.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - meta_train_statistics: list of statistics of train datasets
        - meta_test_statistics: list of statistics of test datasets
      output_dict: Output dict from key to a list of artifacts, currently unused.
      exec_properties: A dict of execution properties
    """
    return None