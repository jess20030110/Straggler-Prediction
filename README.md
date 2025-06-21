# SPF
### This repository contains a Straggler Prediction Framework (SPF), a two-stage pipeline to analyze the Alibaba Cluster Trace 2020 dataset.
1. Stage 1: Duration Prediction: An MLP (Multi-Layer Perceptron) model is trained to predict the execution duration of job instances.
2. Stage 2: Straggler Classification: The predicted duration is then used as a feature to train a RandomForestClassifier that identifies "straggler" instances, those that run for an unusually long time.

## Prerequisite
### Software
- Python 3.7+
- An NVIDIA GPU is recommended for faster training but is not required.
### Python Libraries
- Install the necessary libraries using pip or a conda environment.
```bash
pip install pandas numpy scikit-learn torch imblearn 
```

## Dataset
- This project requires the Alibaba Cluster Trace 2020 dataset, which can be downloaded from https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

## Directory Structure
For the scripts to run correctly, you must create the Preprocess and Result directories. The project should be organized as follows:
``` bash
/path/to/your/project/
|
├── 2020_Trace/
|   └── Dataset/                      # Raw data from Alibaba Trace
|       ├── pai_instance_table/
|       |   └── pai_instance_table.csv
|       └── ... (other raw data folders)
|
├── Preprocess/                       # For intermediate files from initial data prep
|
├── Result/                           # For all final models and key data artifacts
|
└── Scripts/                          # Location for the Python scripts
    ├── merge_dataset.py
    ├── preprocess.py
    ├── 1_train_duration_model.py
    └── 2_classify_stragglers.py
```

## Usage
1. python preprocessing.py
2. python predict_duration&straggler.py
