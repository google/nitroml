# Copyright 2021 Google LLC
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
"""Utils for KubeFlow."""

import os

from absl import logging

try:
  from tfx.orchestration.kubeflow import kubeflow_dag_runner  # pylint: disable=g-import-not-at-top # type: ignore
except ModuleNotFoundError:
  pass


def get_default_kubeflow_dag_runner():
  """Returns the default KubeflowDagRunner with its default metadata config."""

  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)
  logging.info('Using "%s" as  the docker image.', tfx_image)
  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config, tfx_image=tfx_image)

  return kubeflow_dag_runner.KubeflowDagRunner(config=runner_config)
