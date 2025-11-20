# Introduction to MLFlow

MLflow is an open-source platform that helps you track, reproduce, and deploy machine learning models.

It has four main components:

- Tracking – record and query experiments (parameters, metrics, artifacts, etc.)

- Projects – package ML code in a reusable format

- Models – manage model versions and serve them

- Registry – store and manage different model versions

The tracking URI tells MLflow where to log and retrieve your experiment data — essentially, where the MLflow Tracking Server is located. It’s how MLflow knows whether to store your experiment metadata: locally (in files on your machine), or remotely (in a central MLflow server or database).

For permanently removing all experiments (or the `--experiment-ids` option), we can use:
```bash
mlflow gc --backend-store-uri sqlite:////absolute/path/to/mlflow.db tracking-uri=http://127.0.0.1:5000/
```

In addition, most of MLFlow CLI commands need a tracking URI that can be set as a environment variable for the current shell session:

```bash
export MLFLOW_TRACKING_URI=sqlite:////absolute/path/to/mlflow.db 
```

or running the environment variable for a specific command:

```bash
MLFLOW_TRACKING_URI=sqlite:////abs/path/mlflow.db mlflow experiments search --view all
MLFLOW_TRACKING_URI=sqlite:////abs/path/mlflow.db mlflow experiments delete -x 1
```