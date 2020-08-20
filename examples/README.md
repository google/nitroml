# NitroML Examples

Canonical examples of how to use NitroML. These examples are maintained by the NitroML Team in Google Research, and serve as templates to fork and modify.

## Included examples

1.  **titanic_benchmark.py**: A simple AutoML pipeline run on the Titanic dataset from TensorFlow datasets.
2.  **openml_cc18_benchmark.py**: A simple AutoML pipeline run on the [OpenML-CC18](https://www.openml.org/s/99) benchmark suite composed of 72 dataset.
3.  **meta_learning_benchmark.py**: A simple metalearning AutoML pipeline run using a subset of [OpenML-CC18](https://www.openml.org/s/99) benchmark suite.

## Run Nitroml on the Cloud with KubeFlow

- Open the jupyter notebook `nitroml_kubeflow.ipynb`. Be sure to update the `ENDPOINT` in the notebook.
- Update the `GCP_BUCKET_NAME` in `examples/config.py` based on your Google Cloud Project.

## Connect to MLMD database when working on Cloud and Kubeflow

- Step 0: Download and [configure gcloud SDK](https://cloud.google.com/sdk/docs/initializing) using your account.

- Step 1: Configure your cluster with gcloud.
  - Run `gcloud container clusters get-credentials <cluster_name> --zone <cluster-zone> --project <project-id>` from your local machine.
  - `cluster_name` is the name of the kubeflow cluster, `cluster-zone` is the zone of the cluster,  `project-id` is the google cloud project.

- Step 2: Get the port where the gRPC service is running on the cluster
  - `kubectl get configmap metadata-grpc-configmap -o jsonpath={.data}`
  - Use `METADATA_GRPC_SERVICE_PORT` in the next step. The default port used is 8080.

- Step 3: Port forwarding
  - `kubectl port-forward deployment/metadata-grpc-deployment 9898:<METADATA_GRPC_SERVICE_PORT>`

- Troubleshooting
  - If getting error related to Metadata (For examples, Transaction already open). Try restarting the metadata-grpc-service using: `kubectl rollout restart deployment metadata-grpc-deployment`
