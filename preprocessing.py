import pandas as pd

# Define column names
TASK_COLS = ['job_name', 'task_name', 'inst_number', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type']
GROUP_COLS = ['inst_id', 'user', 'gpu_type_spec', 'group', 'workload']
JOB_COLS = ['job_name', 'inst_id', 'user', 'status', 'start_time', 'end_time']
INSTANCE_COLS = ['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id', 'status', 'start_time', 'end_time', 'machine']
MACHINE_METRIC_COLS = ['worker_name', 'machine', 'start_time', 'end_time', 'machine_cpu_iowait', 'machine_cpu_kernel', 'machine_cpu_usr', 'machine_gpu', 'machine_load_1', 'machine_net_receive', 'machine_num_worker', 'machine_cpu']
MACHINE_SPEC_COLS = ['machine', 'gpu_type', 'cap_cpu', 'cap_mem', 'cap_gpu']
SENSOR_COLS = ['job_name', 'task_name', 'worker_name', 'inst_id', 'machine', 'gpu_name', 'cpu_usage', 'gpu_wrk_util','avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count']

# Load data
print("Loading data...")
df_instance = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_instance_table/pai_instance_table.csv", header=None, names=INSTANCE_COLS)
df_task = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_task_table/pai_task_table.csv", header=None, names=TASK_COLS)
df_group = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_group_tag_table/pai_group_tag_table.csv", header=None, names=GROUP_COLS)
df_job = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_job_table/pai_job_table.csv", header=None, names=JOB_COLS)
df_machine_metric = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_machine_metric/pai_machine_metric.csv",header=None, names=MACHINE_METRIC_COLS)
df_machine_spec = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_machine_spec/pai_machine_spec.csv",header=None, names=MACHINE_SPEC_COLS)
df_sensor = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_sensor_table/pai_sensor_table.csv",header=None, names=SENSOR_COLS)

print(f"Instance: {len(df_instance):,} | Task: {len(df_task):,} | Job: {len(df_job):,} | Machine_metric: {len(df_machine_metric):,}| Machine_spec: {len(df_machine_spec):,} | Sensor: {len(df_sensor):,}")

# 1. Filter for Terminated status
print("\nFiltering for Terminated status...")
df_instance = df_instance[df_instance['status'] == 'Terminated']
df_task = df_task[df_task['status'] == 'Terminated']
df_job = df_job[df_job['status'] == 'Terminated']
print(f"Instance: {len(df_instance):,} | Task: {len(df_task):,} | Job: {len(df_job):,}| Machine_metric: {len(df_machine_metric):,}| Machine_spec: {len(df_machine_spec):,} | Sensor: {len(df_sensor):,}")

# First, ensure consistent data types for merge keys
df_machine_spec['machine'] = df_machine_spec['machine'].astype(str)
df_machine_metric['machine'] = df_machine_metric['machine'].astype(str)
df_instance['machine'] = df_instance['machine'].astype(str)

# 2. Merge 7 datasets
print("\nMerging datasets...")
# Step 1: Merge task + job
df_merged = pd.merge(
    df_task, 
    df_job, 
    on='job_name', 
    suffixes=('_task', '_job')
)
print(f"After task+job merge: {len(df_merged):,} rows")

# Step 2: Merge with instance data
# First check common columns
common_cols = set(df_merged.columns) & set(df_instance.columns)
print(f"Common columns with instance: {common_cols}")

df_merged = pd.merge(
    df_merged,
    df_instance,
    on=['job_name', 'task_name', 'inst_id'],  # Now inst_id is merged into one column
    suffixes=('', '_instance')  # Only suffix conflicting non-key columns
)
print(f"\nTotal rows after merging instances: {len(df_merged):,}")

# Step 3: Merge with group data
df_merged = pd.merge(
    df_merged,
    df_group,
    on=['inst_id', 'user'],  # Now inst_id is consistent
    suffixes=('', '_group')
)
print(f"\nTotal rows after merging group: {len(df_merged):,}")

# Step 4: Merge with machine specs
df_merged = pd.merge(
    df_merged,
    df_machine_spec,
    on='machine',
    suffixes=('', '_machine_spec')
)
print(f"\nTotal rows after merging machine spec: {len(df_merged):,}")

# Step 5: Merge with machine metrics
df_merged = pd.merge(
    df_merged,
    df_machine_metric,
    on=['worker_name', 'machine', 'start_time', 'end_time'],
    how='left',
    suffixes=('', '_machine_metric')
)
print(f"\nTotal rows after merging machine metric: {len(df_merged):,}")

# Step 6: Merge with sensor
df_merged = pd.merge(
    df_merged,
    df_sensor[['gpu_name','job_name', 'task_name', 'inst_id', 'worker_name', 'machine']].drop_duplicates(),
    on=['job_name', 'task_name', 'inst_id', 'worker_name', 'machine'],
    how='left'
)
print(f"\nTotal rows after merging gpu_name: {len(df_merged):,}")

df_sensor_agg = df_sensor.groupby(['job_name', 'task_name', 'inst_id', 'worker_name', 'machine']).agg({
    'cpu_usage': 'mean',
    'gpu_wrk_util': 'mean',
    'avg_mem': 'mean',
    'max_mem': 'max',
    'avg_gpu_wrk_mem': 'mean',
    'max_gpu_wrk_mem': 'max',
    'read': 'sum',
    'write': 'sum',
    'read_count': 'sum',
    'write_count': 'sum'
}).reset_index()

df_final = pd.merge(
    df_merged,
    df_sensor,
    on=['job_name', 'task_name', 'inst_id'],
    how='left',
    suffixes=('', '_sensor')
)

# 3. Filtering steps
print("\nApplying group filters...")
# 3.1 Count unique groups
unique_groups = df_final['group'].nunique()

# 3.2 Count group frequencies
group_counts = df_final['group'].value_counts().reset_index()
group_counts.columns = ['group', 'count']

# 3.3 Filter for groups appearing at least 5 times
min_group_count = 5
frequent_groups = group_counts[group_counts['count'] >= min_group_count]['group']
dfa_filtered = df_final[df_final['group'].isin(frequent_groups)]

print(f"\nRows after keeping only groups with ≥{min_group_count} occurrences: {len(dfa_filtered):,}")
print(f"Number of groups remaining: {dfa_filtered['group'].nunique():,}")

# Final output
print("\nFinal filtered data:")
print(f"Columns with merging: {df_final.columns}")

# 4. Column renaming and adding new columns
print("\nRenaming columns and adding new features...")
# Rename start_time_job to submit_time
dfa_filtered = dfa_filtered.rename(columns={'start_time_job': 'submit_time'})
# Calculate the earliest start_time_task for each job
earliest_start_per_job = dfa_filtered.groupby('job_name')['start_time_task'].min().reset_index()
earliest_start_per_job.columns = ['job_name', 'earliest_start_time_task']
# Merge this back with the original dataframe
dfa_filtered = pd.merge(dfa_filtered, earliest_start_per_job, on='job_name')
# Calculate wait_time (earliest start_time_task - submit_time)
dfa_filtered['wait_time'] = (dfa_filtered['earliest_start_time_task'] - dfa_filtered['submit_time'])
# Calculate duration (end_time - start_time)
dfa_filtered['duration'] = (dfa_filtered['end_time'] - dfa_filtered['start_time'])
# Drop the temporary column
dfa_filtered = dfa_filtered.drop(columns=['earliest_start_time_task'])


# 5. Save the filtered and grouped (but not aggregated) data
output_file = "filtered_grouped_data_new.csv"
dfa_filtered.to_csv(output_file, index=False)
print(f"\nSaved the complete filtered and grouped (but not aggregated) data to {output_file}")
