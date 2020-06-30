"""NitroML config.

This file defines environments for nitroml.
"""
import os

USE_KUBEFLOW = True

PIPELINE_NAME = 'nitroml_examples'
GCS_BUCKET_NAME = 'artifacts.nitroml-brain-xgcp.appspot.com'
PIPELINE_ROOT = os.path.join('gs://', GCS_BUCKET_NAME, PIPELINE_NAME)
DOWNLOAD_DIR = os.path.join('gs://', GCS_BUCKET_NAME, 'tensorflow-datasets')