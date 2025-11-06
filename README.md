# MLOps Demo

## Introduction to Docker

**Docker** is a platform for building, running, and shipping applications in a consistent manner. It allows developers to package applications and their dependencies into lightweight, portable **containers**. These containers run consistently across different environments — from a developer’s laptop to production — without worrying about configuration differences.

With Docker, we can avoid some of the common problems when deploying an application in different machines/devices, such as:
- One or more files are missing.
- Software version mismatch on the target machine.
- Different configuration settings between machines (such as environment variables).

### Key Concepts
- A **container** includes the application code, runtime, libraries, and configuration files needed to run it.  
- Containers share the **host operating system’s kernel**, making them much more efficient and faster to start than virtual machines.  
- **Docker Engine** manages containers, handling their creation, execution, and networking.

### Docker vs. Virtual Machines

| Feature | **Docker (Containers)** | **Virtual Machines (VMs)** |
|----------|-------------------------|-----------------------------|
| **Architecture** | Share the host OS kernel | Each VM includes a full OS |
| **Size** | Lightweight (MBs) | Heavy (GBs) |
| **Startup Time** | Seconds | Minutes |
| **Isolation** | Process-level | Full hardware-level |
| **Performance** | Near-native speed | More overhead due to hypervisor |
| **Portability** | Highly portable | Less portable between systems |

---

### Installing Docker

Docker is already included in GitHub Codespace. In case of using a local machine, visit [this link](https://docs.docker.com/get-started/get-docker/). Make sure your computer follow the basic requirements before the installation.

## Introduction to MLFlow

MLflow is an open-source platform that helps you track, reproduce, and deploy machine learning models.

It has four main components:

- Tracking – record and query experiments (parameters, metrics, artifacts, etc.)

- Projects – package ML code in a reusable format

- Models – manage model versions and serve them

- Registry – store and manage different model versions

The tracking URI tells MLflow where to log and retrieve your experiment data — essentially, where the MLflow Tracking Server is located. It’s how MLflow knows whether to store your experiment metadata: locally (in files on your machine), or remotely (in a central MLflow server or database)