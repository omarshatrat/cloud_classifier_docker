# Cloud Model Deployment Project

This repository contains the code and configurations required to deploy a machine learning model that classifies types of clouds. It's set up to use Docker to ensure that the environment is reproducible and that the model can be run consistently across different machines.

## Repository Structure

- `pipeline.py`: **This is the main script that runs the entire machine learning pipeline, from data acquisition to model training and evaluation, and finally uploading the results to AWS S3**.
- `src/`: This directory contains the Python modules that the main pipeline script uses. Each module is responsible for different stages of the pipeline:
  - `acquire_data.py`: Functions for getting data from a URL and constructing/saving the dataset.
  - `eda.py`: Functions to generate data visualizations and save them locally.
  - `generate_features.py`: Functions to generate features used to train and machine learning model.
  - `train_model.py`: Functions to train the machine learning model, evaluate its performance, and save model artifacts.
  - `aws_utils.py`: Utilities for uploading artifacts to AWS S3.
- `config/`: Contains YAML/logging configuration files that control various aspects of the pipeline, such as data sources, model parameters, and AWS settings.
- `artifacts/`: Local directory where artifacts from the pipeline will be saved.
- `data/`: Local directory where data needed for model training and feature generation will be stored.
- `dockerfiles/': Contains Docker files for both the pipeline and unit_testing
  - `Dockerfile.dockerfile`: Defines Docker environment for running the pipeline.
  - `Dockerfile_testing.dockerfile`: Defines the Docker environment specifically for running tests.
- `tests/`: Contains pytest file used for generating features.
  - `generate_features_test.py`: Contains unit testing script for generate_features module.
- `requirements.txt`: Lists dependencies and packages needed to run the Docker file.

# Instructions to Run

### Note about OS
These instructions were developed using Windows 11 Pro and PowerShell. Attempts to provide the corresponding Mac commands have been included where appropriate.

## Cloning the Repository

To clone this repository and start working with it, run the following command:

```bash
git clone https://github.com/omarshatrat/cloud_classifier_docker.git
cd cloud_classifier_docker
```


## Customizing the YAML Configuration File

The YAML configuration file in the `config/` directory controls various parameters of the pipeline. Make changes to the YAML file to match your project requirements and AWS configuration. 

### Make sure to change the bucket name within the config file to an existing bucket name within your S3 account.

## Building and Running the Docker Container
Since everything will be run in docker, users don't need to install anything except for docker itself. Before building and running the Docker containers, you must have Docker installed on your system. Visit [Docker's official website](https://www.docker.com/get-started) for installation instructions tailored to your operating system.

To authenticate and configure your AWS credentials, run:

```bash
aws configure sso --profile default
```

Once you have set up your desired configurations, log into AWS using:

```bash
aws sso login
```

The command should take you to an external website where you can complete the authentication process.


To build the Docker container for the main pipeline, run:

### Windows command:

```bash
docker build -f .\dockerfiles\Dockerfile.dockerfile -t <DESIRED IMAGE NAME> .
```

### Mac command:

```bash
docker build -f ./dockerfiles/Dockerfile.dockerfile -t <DESIRED IMAGE NAME> .
```

If you encounter any errors building the image, consider switching to a different CLI such as PowerShell.

To run the pipeline:

```bash
docker run -v ${HOME}/.aws/:/root/.aws/:ro -v ${PWD}/artifacts:/app/artifacts -v ${PWD}/data:/app/data --name <DESIRED CONTAINER NAME> <DESIRED IMAGE NAME>
```
Note that you have to configure and login to your AWS profile locally before running this command.
After running this, the container will build the model, and upload the model artifact to the s3 bucket specified in the YAML file.

## Building and Running Tests with Docker

To build the Docker container for running tests:

### Windows command:

```bash
docker build -f .\dockerfiles\Dockerfile_testing.dockerfile -t <DESIRED IMAGE NAME - DIFFERENT FROM ABOVE> .
```

### Mac command:

```bash
docker build -f ./dockerfiles/Dockerfile_testing.dockerfile -t <DESIRED IMAGE NAME - DIFFERENT FROM ABOVE> .
```

To run the tests:

```bash
docker run -v ${HOME}/.aws/:/root/.aws/:ro --name <DESIRED CONTAINER NAME - DIFFERENT FROM ABOVE> <DESIRED IMAGE NAME - DIFFERENT FROM ABOVE>
```

Once the pipeline has been run, you should find that the appropriate artifacts have been both uploaded to S3 and saved locally in the locations aforementioned.
