import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set the tracking URI to a local database URI (e.g., sqlite:///mlflow.db).
# This is recommended option for quickstart and local development.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MLFlow Quickstart")

# Training Code Example

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# To enable autologging for scikit-learn, uncomment the following code:
# mlflow.sklearn.autolog()

# # Just train the model normally
# lr = LogisticRegression(**params)
# lr.fit(X_train, y_train)

# To log a model and medata manually, uncomment the following code:
with mlflow.start_run():
    # Log model
    mlflow.log_params(params)

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    # Predict on the test set, compute and log the loss metric
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Optional: Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

# Access MLflow UI at http://localhost:5000.
# To start the MLflow UI, run the following command in your terminal:
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
