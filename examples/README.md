# NitroML Examples

Canonical examples of how to use NitroML. These examples are maintained by the NitroML Team in Google Research, and serve as templates to fork and modify.

## Included examples

1.  **titanic_benchmark.py**: A simple AutoML pipeline run on the Titanic dataset from TensorFlow datasets.
2.  **openml_cc18_benchmark.py**: A simple AutoML pipeline run on the [OpenML-CC18](https://www.openml.org/s/99) benchmark suite composed of 72 dataset.
3.  **meta_learning_benchmark.py**: A simple metalearning AutoML pipeline run using a subset of [OpenML-CC18](https://www.openml.org/s/99) benchmark suite. See below on visualizing tuner results for metalearning examples.

## Run Nitroml on the Cloud with KubeFlow

- Open the jupyter notebook `nitroml_kubeflow.ipynb` and follow the steps.
  - To open the jupyter notebook: execute `jupyter notebook` from `examples/` dir.
  - Set the Kubeflow `ENDPOINT` in the `examples/config.py`.
  - Set the `GCS_BUCKET_NAME` in the `examples/config.py`.
  - For most up-to-date information on finding your `ENDPOINT` and `GCS_BUCKET_NAME` please see this [tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines).
- Kubeflow uses TFX Docker image when running pipelines. You can find latest images on [docker hub](https://hub.docker.com/r/tensorflow/tfx/tags). If you want to change the TFX version, just change the `TFX_IMAGE` variable in `examples/config.py` based on images uploaded on [docker hub](https://hub.docker.com/r/tensorflow/tfx/tags).

## Connect to MLMD database when working on Cloud and Kubeflow

- Step 0: Download and [configure gcloud SDK](https://cloud.google.com/sdk/docs/initializing) using your account.

- Step 1: Configure your cluster with gcloud.

  - Run `gcloud container clusters get-credentials <cluster_name> --zone <cluster-zone> --project <project-id>` from your local machine.
  - `cluster_name` is the name of the kubeflow cluster, `cluster-zone` is the zone of the cluster, `project-id` is the google cloud project.

- Step 2: Get the port where the gRPC service is running on the cluster

  - `kubectl get configmap metadata-grpc-configmap -o jsonpath={.data}`
  - Use `METADATA_GRPC_SERVICE_PORT` in the next step. The default port used is 8080.

- Step 3: Port forwarding

  - `kubectl port-forward deployment/metadata-grpc-deployment 9898:<METADATA_GRPC_SERVICE_PORT>`

- Step 4: Open and run the Jupyter Notebook`kubeflow_metadata_example.ipynb` to visualize the benchmark results.

- Troubleshooting
  - If getting error related to Metadata (For examples, Transaction already open). Try restarting the metadata-grpc-service using: `kubectl rollout restart deployment metadata-grpc-deployment`

## Visualize tuner progress (Works in conjunction with `meta_learning_examples.py`)

- Follow the above steps on connecting to the MLMD database (if working with Kubeflow)
- Open and run the Jupyter Notebook `visualize_tuner_plots.ipynb`
