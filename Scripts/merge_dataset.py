import pandas as pd

# 1. Data Loading and Preprocessing
print("1. Loading and preprocessing data...")
df_merged = pd.read_csv("/mnt/newdisk/jess/Alibaba2020/Preprocess/instance_merged.csv")
print(f"Loaded {len(df_merged):,} rows")

print(f"Number of groups before filter: {df_merged['group'].nunique():,}")
print("\nFiltering for groups with at least 5 occurrences...")
group_counts = df_merged.groupby('group')['inst_name'].transform('count')
df = df_merged[group_counts >= 5].copy()

print(f"\nRows after keeping only groups with ≥5 occurrences: {len(df):,}")
print(f"Number of groups remaining: {df['group'].nunique():,}")

# --- Drop rows with missing 'end_time' or 'duration' or 'submit_time'---
print(f"\nOriginal rows in filtered dataset: {len(df):,}")
print("Dropping rows with null values in 'end_time' or 'duration' or 'submit_time'...")
df = df.dropna(subset=['end_time', 'duration', 'submit_time'])
print(f"Rows after dropping NaNs: {len(df):,}")

# Handle missing GPU data
drop_cols = [
    'machine_cpu', 'machine_cpu_iowait', 
    'machine_num_worker', 'machine_cpu_kernel', 
    'machine_load_1', 'machine_gpu', 'machine_cpu_usr',
    'machine_net_receive', 'workload', 'gpu_type_spec'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

gpu_nan_cols = [
    'plan_gpu','avg_gpu_wrk_mem', 
    'max_gpu_wrk_mem', 'gpu_wrk_util'
]
df[gpu_nan_cols] = df[gpu_nan_cols].fillna(0)
print(f"Number of rows after fill 0: {len(df)}")

moderate_nan_cols = [
    'cpu_usage', 'avg_mem', 'max_mem',
    'read_count', 'write_count', 'read', 'write', 'gpu_name'
]
df = df.dropna(subset=moderate_nan_cols)
print(f"Number of rows after dropping NaN: {len(df)}")
print(df.shape)

# Display columns with NaN values in the final filtered dataset
print("\nColumns with NaN values in the filtered dataset:")
nan_counts = df.isnull().sum()
nan_columns = nan_counts[nan_counts > 0]
if nan_columns.empty:
    print("No columns with NaN values.")
else:
    print(nan_columns)

# Save the filtered dataset
output_file_filtered = "/mnt/newdisk/jess/Alibaba2020/Preprocess/instance_preprocessed.csv"
df.to_csv(output_file_filtered, index=False)
print(f"\nSaved filtered dataset to {output_file_filtered}")
