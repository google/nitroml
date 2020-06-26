# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define KubeflowDagRunner to run the pipeline using Kubeflow.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from absl import logging
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from nitroml import nitroml
import titanic_benchmark
import config

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  pipeline_root = os.path.join('gs://', config.GCS_BUCKET_NAME, 'tfx_pipeline_output')
  download_dir = os.path.join('gs://', config.GCS_BUCKET_NAME, 'tensorflow-datasets')
  nitroml.main(pipeline_name=config.PIPELINE_NAME, pipeline_root=pipeline_root, data_dir=download_dir, kubeflow=True)