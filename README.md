# SPF
### This repository contains a Straggler Prediction Framework (SPF), a two-stage pipeline to analyze the Alibaba Cluster Trace 2020 dataset.
- Stage 1: Duration Prediction: An MLP (Multi-Layer Perceptron) model is trained to predict the execution duration of job instances.
- Stage 2: Straggler Classification: The predicted duration is then used as a feature to train a RandomForestClassifier that identifies "straggler" instances, those that run for an unusually long time.

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
- You need the following CSV files from the Alibaba 2020 cluster trace dataset:
```bash
2020_Trace/Dataset/
├── pai_instance_table/pai_instance_table.csv
├── pai_task_table/pai_task_table.csv
├── pai_group_tag_table/pai_group_tag_table.csv
├── pai_job_table/pai_job_table.csv
├── pai_machine_metric/pai_machine_metric.csv
├── pai_machine_spec/pai_machine_spec.csv
└── pai_sensor_table/pai_sensor_table.csv
```

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
├── Preprocess/                       # Intermediate and processed files
|
├── Result/                           # Model outputs and results
|
└── Scripts/                          # Location for the Python scripts
    ├── merge_dataset.py
    ├── preprocess.py
    ├── predict_durations_MLP.py
    └── classify_stragglers.py
```

## Execution Workflow
### The project involves a sequential, four-step process. Run the scripts in the following order.
#### Step 1: Merge Raw Datasets
##### This script combines the various raw data tables into a single CSV file.
``` bash
python merge_dataset.py
```
- Input: Raw .csv files from 2020_Trace/Dataset/.
- Output: Preprocess/instance_merged.csv.

#### Step 2: Preprocess Merged Data
##### This script cleans the merged data, handles missing values, and filters records to create a dataset ready for modeling.
``` bash
python preprocess.py
```
- Input: Preprocess/instance_merged.csv.
- Output: Preprocess/instance_preprocessed.csv.

#### Step 3: Train Duration Prediction Model
##### This script performs all deep learning tasks. It trains the MLP model, uses the trained model to generate predictions for the entire dataset, and saves all its key outputs to the Result folder.
```bash
python predict_durations_MLP.py
```
- Input: Preprocess/instance_preprocessed.csv.
- Outputs:
   1. A trained PyTorch model saved as Result/mlp_model_hiddenScale6.pth.
   2. An augmented data file saved as Result/predictions_for_classification.csv. This file contains all original features plus the new predicted_duration column, absolute_error column, absolute_percentage_error column, and a split column.

#### Step 4: Classify Straggler Instances
##### This script loads the augmented data from Step 3 and focuses entirely on training a RandomForestClassifier to identify straggler instances.
```bash
python classify_stragglers.py
```
- Input: Result/predictions_for_classification.csv.
- Output:
   1. A final results file saved as Result/final_straggler_classification_results.csv.
   2. Classifier performance metrics (F1 Score, Precision, and Recall) are printed to the console.
