# NitroML

## Starter Project

### Directory structure

- DAG code is in `simple_pipeline.py`.
- The `datasets/dataset.py` dir has the code to download OpenML Datasets.

### To download the dataset:
- Download OpenML datasets using `python download_OpenML.py --download`

Steps to use Airflow to run the pipeline:

- Create & activate a Python3 conda environment.
- `cd` to nitroml dir.
- Run `setup.sh` - This will install all the requirements (including tfx & airflow  dependencies)
- Execute `chmod +x -R ~/airflow/dags/datasets`
- Execute `chmod +x -R ~/airflow/dags/auto_transform_component`
- start a tmux and activate the conda environment: Run: `airflow webserver -p 9090`
- start another tmux and activate the conda environment: Run: `airflow scheduler`
- In your browser, open `localhost:9090` to view the airflow console where we can trigger the pipeline.

*****
The above setup assumes that the codebase lives in the <HOME_DIR>. We will have to change the module paths in `simple_pipeline.py` if we change the location of codebase.