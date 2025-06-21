import pandas as pd
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import TargetEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def print_step(message):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Track total execution time
total_start = time.time()

# 0. GPU Setup
print_step("0. Initializing GPU...")
GPU_AVAILABLE = False
try:
    import cudf
    from cuml.ensemble import RandomForestRegressor  # Using RF as single-tree proxy
    import cupy as cp
    GPU_AVAILABLE = True
    print("⚡ RAPIDS cuML detected - Using GPU acceleration (with RF proxy)")
except ImportError as e:
    from sklearn.tree import DecisionTreeRegressor
    print(f"⚠️ RAPIDS not found - Falling back to CPU\nImport error: {e}")

# 1. Data Loading
print_step("1. Loading data...")
df = pd.read_csv("filtered_grouped_data.csv")
print(f"Loaded {len(df):,} rows")
print("Columns:", df.columns.tolist())

# # Get data types and non-null counts
# df_info = df.dtypes.reset_index()
# df_info.columns = ['Column', 'Data Type']
# print(df_info)
# # Count NaN values in each column
# nan_counts = df.isnull().sum().reset_index()
# nan_counts.columns = ['Column', 'NaN Count']
# print(nan_counts.sort_values(by='NaN Count', ascending=False))

# Handle missing GPU data
drop_cols = [
    'machine_cpu', 'machine_cpu_iowait', 'workload', 
    'machine_num_worker', 'machine_cpu_kernel', 
    'machine_load_1', 'machine_gpu', 'machine_cpu_usr',
    'machine_net_receive'
]
df = df.drop(columns=drop_cols)

moderate_nan_cols = [
    'cpu_usage', 'avg_mem', 'avg_gpu_wrk_mem', 
    'max_gpu_wrk_mem', 'gpu_wrk_util', 'max_mem',
    'read_count', 'write_count', 'read', 'write',
    'plan_gpu'
]
# Fill with 0 if NaN indicates "no usage" (e.g., GPU metrics for CPU-only tasks)
df[moderate_nan_cols] = df[moderate_nan_cols].fillna(0)

df['gpu_type'] = df['gpu_type'].fillna('NO_GPU')
if 'gpu_type_spec' in df.columns:
    df['gpu_type'] = df['gpu_type'].combine_first(df['gpu_type_spec'])
    df = df.drop(columns=['gpu_type_spec'])
if 'gpu_type_machine_spec' in df.columns:
    df['gpu_type'] = df['gpu_type'].combine_first(df['gpu_type_machine_spec'])
    df = df.drop(columns=['gpu_type_machine_spec'])

# 2. Feature Engineering
print_step("2. Preparing features...")
df_encoded = pd.get_dummies(df, columns=['gpu_type'], prefix='gpu_')
# df_encoded = pd.get_dummies(df_encoded, columns=['task_name'], prefix='task_')
# df_encoded = pd.get_dummies(df_encoded, columns=['workload'], prefix='workload_')

print("Encoding high-cardinality variables...")
# Check if GPU processing is available
using_gpu = False
try:
    import cudf
    # Try to use GPU for data processing
    df_gpu = cudf.DataFrame.from_pandas(df_encoded)
    using_gpu = True
    print("Successfully loaded data to GPU")
except (ImportError, Exception) as e:
    print(f"GPU processing unavailable: {e}. Using CPU instead.")
    using_gpu = False

# Function for mean target encoding that works with both pandas and cudf
def mean_target_encode(data, cat_column, target_column, alpha=5):
    global_mean = float(data[target_column].mean())
    
    if using_gpu:
        # GPU version
        # Group by the categorical column and calculate mean and count
        agg = data.groupby(cat_column)[target_column].agg(['mean', 'count'])
        
        # Convert to pandas for easier handling if it's a cudf DataFrame
        if hasattr(agg, 'to_pandas'):
            agg = agg.to_pandas()
            
        # Calculate smoothed means
        encoded_values = {}
        for idx, row in agg.iterrows():
            encoded_values[idx] = (row['count'] * row['mean'] + alpha * global_mean) / (row['count'] + alpha)
        
        # Apply the mapping
        if hasattr(data, 'to_pandas'):
            # For cudf Series, convert to pandas, map, and convert back
            pd_series = data[cat_column].to_pandas()
            encoded = pd_series.map(encoded_values).fillna(global_mean)
            return cudf.Series(encoded)
        else:
            # For pandas Series
            return data[cat_column].map(encoded_values).fillna(global_mean)
    else:
        # CPU version
        agg = data.groupby(cat_column)[target_column].agg(['mean', 'count'])
        agg['encoded'] = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
        encoding_map = agg['encoded'].to_dict()
        return data[cat_column].map(encoding_map).fillna(global_mean)

# Target encoding for high-cardinality variables
print("Target encoding user categories...")
user_counts = df_encoded['user'].value_counts()
print(f"User categories: {len(user_counts)}")
df_encoded['user_encoded'] = mean_target_encode(df_encoded, 'user', 'duration')

print("Target encoding group categories...")
group_counts = df_encoded['group'].value_counts()
print(f"Group categories: {len(group_counts)}")
df_encoded['group_encoded'] = mean_target_encode(df_encoded, 'group', 'duration')

# Clean up intermediate columns
if 'user_label' in df_encoded.columns and 'group_label' in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=['user_label', 'group_label'])

# Convert back to pandas if needed for further processing
if using_gpu and hasattr(df_encoded, 'to_pandas'):
    df_encoded = df_encoded.to_pandas()

print("Feature engineering completed!")

features = ['user_encoded', 'plan_gpu', 'plan_mem', 'plan_cpu', 'cap_cpu',
            'cap_gpu', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
            'wait_time', 'inst_number', 'cap_mem','avg_gpu_wrk_mem', 
            'max_gpu_wrk_mem', 'read', 'write','read_count', 'write_count'] + \
           [col for col in df_encoded.columns if col.startswith('gpu_')] + \
           ['group_encoded']
target = "duration"

# 3. Data Preparation
print_step("3. Preparing data...")
if GPU_AVAILABLE:
    # Convert to cuDF and ensure numeric types
    gdf = cudf.DataFrame()
    for col in features + [target]:
        gdf[col] = cudf.Series(df_encoded[col].fillna(0).astype('float32'))
    X = gdf[features]
    y = gdf[target]
else:
    X = df_encoded[features].fillna(0).astype('float32')
    y = df_encoded[target].astype('float32')

# 4. Train-Test Split
print_step("4. Splitting data...")
if GPU_AVAILABLE:
    from cuml.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Explicit conversion to cupy arrays for cuml
    X_train = cp.asarray(X_train.values)
    X_test = cp.asarray(X_test.values)
    y_train = cp.asarray(y_train.values)
    y_test = cp.asarray(y_test.values)
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
print_step("5. Training model...")
train_start = time.time()

if GPU_AVAILABLE:
    model = RandomForestRegressor(
        n_estimators=1,
        max_depth=10,
        random_state=42,
        n_streams=1  # Added to address reproducibility warning
    )
    model.fit(X_train, y_train)
else:
    model = DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

print(f"Trained in {time.time() - train_start:.2f} seconds")

# 6. Evaluation
print_step("6. Evaluating model...")
if GPU_AVAILABLE:
    predictions = model.predict(X_test).get()
    y_test_np = y_test.get()
else:
    predictions = model.predict(X_test)
    y_test_np = y_test.values

# Calculate MAE
mae = mean_absolute_error(y_test_np, predictions)
print(f"Model MAE: {mae:.2f} seconds")

# Calculate percentage error with zero handling
valid_mask = y_test_np != 0
if np.any(valid_mask):
    percentage_error = np.mean(np.abs((y_test_np[valid_mask] - predictions[valid_mask]) / 
                           y_test_np[valid_mask])) * 100
    print(f"Mean Absolute Percentage Error (non-zero): {percentage_error:.2f}%")
else:
    print("Warning: All true durations are zero - cannot calculate percentage error")


# 7. Generate Output
print_step("7. Preparing output...")
if GPU_AVAILABLE:
    X_all = cp.asarray(gdf[features].values)
    df_encoded['predicted_duration'] = model.predict(X_all).get()
    # Copy predictions back to the original dataframe
    df['predicted_duration'] = df_encoded['predicted_duration']
else:
    df_encoded['predicted_duration'] = model.predict(df_encoded[features])
    # Copy predictions back to the original dataframe
    df['predicted_duration'] = df_encoded['predicted_duration']

# Calculate percentage error for all predictions
df['percentage_error'] = ((df['duration'] - df['predicted_duration']) / df['duration']) * 100

# Straggler analysis
grouping_cols = ['job_name', 'task_name']
df['avg_duration'] = df.groupby(grouping_cols)['duration'].transform('mean')
df['straggler_threshold'] = df['avg_duration'] * 1.5
df['is_straggler'] = df['duration'] > df['straggler_threshold']
df['pred_avg_duration'] = df.groupby(grouping_cols)['predicted_duration'].transform('mean')
df['pred_straggler_threshold'] = df['pred_avg_duration'] * 1.5
df['pred_is_straggler'] = df['predicted_duration'] > df['pred_straggler_threshold']

output_columns = [
    'inst_id', 'user', 'plan_gpu', 'plan_cpu', 'plan_mem', 'task_name',
    'inst_number', 'duration', 'predicted_duration', 'percentage_error',
    'job_name', 'submit_time', 'group', 'gpu_type', 'wait_time',
    'is_straggler', 'pred_is_straggler'
]

output_df = df[output_columns].rename(columns={'inst_id': 'job_id'})
output_file = "predictions_with_errors.csv"
output_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
print("="*50)
print("Final output columns:", output_df.columns.tolist())
