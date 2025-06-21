import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import os

def print_step(message):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")

# --- Configuration ---
BASE_PATH = "/mnt/newdisk/jess/Alibaba2020"
PREPROCESS_PATH = f"{BASE_PATH}/Preprocess"
RESULT_PATH = f"{BASE_PATH}/Result"
# This is the single input file for this script
INPUT_CSV_PATH = f"{RESULT_PATH}/predictions_for_classification.csv"
STRAGGLER_PERCENTILE = 0.90

# --- Main Execution ---
total_start = time.time()

# 1. Load Data with Predictions
print_step(f"1. Loading data from {INPUT_CSV_PATH}...")
try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit()
print(f"Loaded {len(df):,} rows with pre-computed predictions.")


# 2. Define Straggler Labels
print_step("2. Defining straggler labels using the training set split...")
train_data = df[df['split'] == 'train'].copy()
group_thresholds = train_data.groupby('group')['duration'].quantile(STRAGGLER_PERCENTILE)
global_threshold = train_data['duration'].quantile(STRAGGLER_PERCENTILE)

df['duration_threshold'] = df['group'].map(group_thresholds).fillna(global_threshold)
df['is_straggler'] = (df['duration'] > df['duration_threshold']).astype(int)

print(f"Train stragglers: {df[df['split'] == 'train']['is_straggler'].sum()} ({df[df['split'] == 'train']['is_straggler'].mean()*100:.2f}%)")
print(f"Test stragglers: {df[df['split'] == 'test']['is_straggler'].sum()} ({df[df['split'] == 'test']['is_straggler'].mean()*100:.2f}%)")


# 3. Train RandomForest Straggler Classifier
print_step("3. Training RandomForest Straggler Classifier with SMOTE...")
df['pred_duration_per_cpu'] = df['predicted_duration'] / df['plan_cpu'].replace(0, 1)

# Define features for the classifier
# Note: Most of these features already exist in the loaded CSV
RandomForest_features = [
    'predicted_duration', 'plan_gpu', 'plan_cpu', 'plan_mem', 'cpu_usage', 'gpu_wrk_util',
    'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'days_since_start', 'avg_mem', 'max_mem',
    'user_encoded', 'group_encoded', 'machine_encoded', 'pred_duration_per_cpu',
    'read_count', 'write_count', 'read', 'write'
] + [col for col in df.columns if col.startswith('gpu_type_') or col.startswith('task_name_')]
RandomForest_features = [f for f in RandomForest_features if f in df.columns]
print(RandomForest_features)

# Prepare train/test sets using the 'split' column
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

X_train_rf = train_df[RandomForest_features].fillna(0).values
y_train_rf = train_df['is_straggler'].values
X_test_rf = test_df[RandomForest_features].fillna(0).values
y_test_rf = test_df['is_straggler'].values

# Define classification pipeline
pipeline = ImbPipeline([
    ('feature_selection', SelectKBest(f_classif, k=36)),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1))
])

# 4. Fit the pipeline and evaluate
print_step("4. Fitting the pipeline and evaluating on the test set...")
pipeline.fit(X_train_rf, y_train_rf)
y_test_pred = pipeline.predict(X_test_rf)

print(f"\nTest F1 Score: {f1_score(y_test_rf, y_test_pred):.4f}")
print(f"Test Precision: {precision_score(y_test_rf, y_test_pred):.4f}")
print(f"Test Recall: {recall_score(y_test_rf, y_test_pred):.4f}")

# 5. Save Final Results
print_step("5. Saving final classification results...")
os.makedirs(RESULT_PATH, exist_ok=True)
test_df['rf_straggler_prediction'] = y_test_pred

results_columns = [
    'inst_name', 'job_name', 'duration', 'predicted_duration', 'duration_threshold',
    'is_straggler', 'rf_straggler_prediction', 'absolute_error', 'absolute_percentage_error'
]
final_results = test_df[results_columns].copy()
output_path = f'{RESULT_PATH}/final_straggler_classification_results.csv'
final_results.to_csv(output_path, index=False)

print(f"Final classification results for the test set saved to {output_path}")
print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
