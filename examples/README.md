# NitroML Examples

Canonical examples of how to use NitroML. These examples are maintained by the NitroML Team in Google Research, and serve as templates to fork and modify.

## Included examples

1.  **titanic_benchmark.py**: A simple AutoML pipeline run on the Titanic dataset from TensorFlow datasets.
2.  **openml_cc18_benchmark.py**: A simple AutoML pipeline run on the [OpenML-CC18](https://www.openml.org/s/99) benchmark suite composed of 72 dataset.
3.  **meta_learning_benchmark.py**: A simple metalearning AutoML pipeline run using a subset of [OpenML-CC18](https://www.openml.org/s/99) benchmark suite.

## Run Nitroml on the Cloud with KubeFlow

- Open the jupyter notebook `nitroml_kubeflow.ipynb`. Be sure to update the `ENDPOINT` in the notebook.
- Update the `GCP_BUCKET_NAME` in `examples/config.py` based on your Google Cloud Project.
