[![CI](https://github.com/JTorres258/mlops-demo/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/JTorres258/mlops-demo/actions/workflows/ci_pipeline.yml)

# 🐶 MLOps Demo – Dog Breed Classification (TensorFlow + FastAPI + Docker + MLflow)

This project demonstrates a full MLOps workflow for a deep learning image classification system using:

- **TensorFlow 2.20 (GPU)**
- **FastAPI** for serving the model
- **Docker** (separate images for training and API)
- **VS Code Dev Containers** for reproducible development
- **MLflow** for experiment tracking
- **Optuna** for hyperparameter optimization

The model is trained on the **Stanford Dogs** dataset (120 dog breeds).

---

# 🚀 Features

- GPU-accelerated training using Dev Containers  
- Experiment tracking with MLflow  
- Hyperparameter tuning with Optuna  
- Evaluation metrics: accuracy, MCC, ROC, PR, confusion matrix  
- Per-class metrics saved as JSON  
- FastAPI server for inference  
- Clean separation of training and serving environments  
- Docker-based deployment  

---

# 📁 Project Structure

mlops-demo/
│
├── app/
│   ├── api/               # FastAPI inference application
│   └── train/             # Training, evaluation, dataset loading
│
├── models/                # Saved trained models (.keras files)
│
├── configs/               # Training configs (YAML)
│
├── .devcontainer/         # VS Code GPU development environment
│   └── devcontainer.json
│
├── Dockerfile.train       # GPU training/dev image
├── Dockerfile.api         # FastAPI runtime image
│
├── requirements_train.txt
├── requirements_api.txt
│
├── README.md
└── ...

---

# 🧠 Two-Environment Architecture

This project uses **two separate Docker images**, each designed for a different purpose:

| Purpose | Dockerfile | Runs In | Used For |
|--------|-------------|---------|----------|
| **Training / Development** | `Dockerfile.train` | VS Code Dev Container | Training, evaluation, MLflow, Optuna |
| **API / Serving** | `Dockerfile.api` | Standalone Docker Container | FastAPI inference server |

---

# 🏋️‍♂️ Training & Development Environment (VS Code Dev Container)

### 🎯 Goal
Develop, train, and evaluate models in a **reproducible GPU-enabled environment**.

### 📦 Files

#### `Dockerfile.train`

FROM tensorflow/tensorflow:2.20.0-gpu
WORKDIR /workspace

COPY requirements_train.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install --ignore-installed -r requirements_train.txt

CMD ["bash"]

#### `requirements_train.txt`

mlflow
scikit-learn
optuna
tensorflow_datasets
pillow
pyyaml

#### `.devcontainer/devcontainer.json`
Tells VS Code to open the project inside the GPU training container.

### ▶ How to use (VS Code)

1. Open the project in VS Code  
2. Press: **F1 → “Dev Containers: Reopen in Container”**  
3. VS Code builds and enters the GPU container  
4. Run training:

python -m app.train.main

5. Run evaluation:

python -m app.train.evaluate

This environment provides:
- TensorFlow GPU  
- CUDA/cuDNN  
- MLflow  
- Optuna  
- Full reproducibility  

---

# 🚀 API / Model Serving (FastAPI)

### 🎯 Goal
Serve predictions through a lightweight, production-ready FastAPI container.

### 📦 Files

#### `Dockerfile.api`

FROM python:3.11-slim
WORKDIR /app

COPY requirements_api.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements_api.txt

COPY app ./app
COPY models ./models
COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#### `requirements_api.txt`

fastapi
uvicorn[standard]
tensorflow==2.20.0
pillow
python-multipart
pydantic
pyyaml

### ▶ Build the API image

docker build -t mlops-demo-api -f Dockerfile.api .

### ▶ Run the API service

docker run --rm -p 8000:8000 mlops-demo-api

Then open:

- http://localhost:8000/health
- http://localhost:8000/docs

### 🔥 Hot reload (API development)

docker run --rm -p 8000:8000 -v %cd%:/app mlops-demo-api

---

# 📊 Evaluation Outputs

- ConfusionMatrix.png  
- ROC.png  
- PrecisionRecall.png  
- classification_report.txt  
- per_class_metrics.json  

---

# 🧪 MLflow UI

mlflow ui --backend-store-uri sqlite:///mlflow.db

Open:

http://localhost:5000

---

# 🔧 Hyperparameter Optimization (Optuna)

python -m app.train.tune

---

# 📦 Local Installation (without Docker)

pip install -r requirements_train.txt
pip install -r requirements_api.txt

---

# 🧹 Development Tools

black  
pylint  
pytest  
pytest-cov  

---

# 🏁 Summary

- Train inside GPU devcontainer  
- Serve model in FastAPI Docker container  
- Track experiments with MLflow  
- Tune with Optuna  
- Generate metrics & visualizations  

---

# 📄 License

MIT

- Hyperparameter tuning
- Dataset handling
- Evaluation & metrics
- MLflow experiment tracking



## Dataset

The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. There are 20,580 images, out of which 12,000 are used for training and 8580 for testing. Class labels and bounding box annotations are provided for all the 12,000 images.

**Source: https://www.tensorflow.org/datasets/catalog/stanford_dogs**


```pgsql
┌────────────────────────────────────────────────────────────┐
│                 Your Windows VS Code                       │
│                (edit files normally)                       │
│                                                            │
│   ┌────────────────────────────────────────────────────┐   │
│   │     Training Devcontainer (TF GPU)                 │   │
│   │  - Controlled by .devcontainer/devcontainer.json   │   │
│   │  - Uses Dockerfile.train                           │   │
│   │  - Where your training happens                     │   │
│   └────────────────────────────────────────────────────┘   │
│                                                            │
│   ┌────────────────────────────────────────────────────┐   │
│   │      API Container (FastAPI runtime)               │   │
│   │  - Controlled by Dockerfile.api                    │   │
│   │  - NOT a devcontainer                              │   │
│   │  - You run it with docker run / compose            │   │
│   └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘

```

