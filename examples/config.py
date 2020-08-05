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
"""NitroML config.

This file defines environments for nitroml.
"""
import os

USE_KUBEFLOW = True

PIPELINE_NAME = 'examples'
GCS_BUCKET_NAME = 'artifacts.nitroml-brain-xgcp.appspot.com'
PIPELINE_ROOT = os.path.join('gs://', GCS_BUCKET_NAME, PIPELINE_NAME)
TF_DOWNLOAD_DIR = os.path.join('gs://', GCS_BUCKET_NAME, 'tensorflow-datasets')
OTHER_DOWNLOAD_DIR = os.path.join('gs://', GCS_BUCKET_NAME, 'other-datasets')
ENDPOINT = '38070e0315a0e15-dot-us-east1.pipelines.googleusercontent.com'
TFX_IMAGE = 'tensorflow/tfx:0.23.0.dev20200716'

