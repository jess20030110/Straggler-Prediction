# SPF
## This repository contains a two-stage pipeline for analyzing job execution data and predicting job durations with straggler detection capabilities.

## Prerequisite
- Preprocessing:
  - pandas
- Training:
  - pip install pandas numpy scikit-learn torch torchvision
  - pip install cuml cudf  # For GPU-accelerated KNN (optional)
  - pip install imbalanced-learn matplotlib

## Dataset
- Alibaba cluster-trace-gpu-v2020
- https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

## Usage
1. python preprocessing.py
2. python predict_duration&straggler.py
