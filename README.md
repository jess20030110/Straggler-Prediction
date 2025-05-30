# SPF
### This repository contains SPF, a two-stage pipeline for analyzing job execution data and predicting job durations with straggler detection capabilities. SPF (Straggler Prediction Framework) provides a comprehensive solution for analyzing large-scale job execution data from distributed computing environments. It combines data preprocessing with advanced machine learning techniques to predict job execution times and identify performance anomalies (stragglers).

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
