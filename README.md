# Alzheimers Prediction Pipeline

This project automates the process of testing multiple machine learning methods on a dataset that tracks Alzheimers cases throughout 20 countries.

## Using the Project:

### 1) Clone the Repository

Run the following commands to clone the repository and enter the root directory:
```
git clone https://github.com/jfiengo/AlzheimerPrediction.git
cd AlzheimerPrediction
```

### 2) Create a Virtual Environment

Run the following commands to create and start your virtual environment:
```
python -m venv myenv
source myenv/bin/activate
python -m pip install -r requirements.txt
```

### 3) Data Retrieval

Run the following command to pull the data from the remote:
```
dvc pull
```

### 4) Run the Pipeline

Within your virtual environment, run the following command to run the ML Pipeline:
```
python -m main
```

### 5) Expose the API

Within your virtual environment, run the following command to expose the API on port 5000:
```
python -m app
```

### 6) Exposing API via Docker

*This step is OPTIONAL*

Before proceeding, ensure you have Docker desktop running on your machine.
To build and run a Docker container for this application, run the following commands from the root directory of this project:
```
docker build -t alzheimer-prediction .
docker run -p 5000:5000 alzheimer-prediction
```