import pandas as pd
import numpy as np

TASK_COLS = ['job_name', 'task_name', 'inst_number', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type']
GROUP_COLS = ['inst_id', 'user', 'gpu_type_spec', 'group', 'workload']
JOB_COLS = ['job_name', 'inst_id', 'user', 'status', 'start_time', 'end_time']
INSTANCE_COLS = ['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id', 'status', 'start_time', 'end_time', 'machine']
MACHINE_METRIC_COLS = ['worker_name', 'machine', 'start_time', 'end_time', 'machine_cpu_iowait', 'machine_cpu_kernel', 'machine_cpu_usr', 'machine_gpu', 'machine_load_1', 'machine_net_receive', 'machine_num_worker', 'machine_cpu']
MACHINE_SPEC_COLS = ['machine', 'gpu_type', 'cap_cpu', 'cap_mem', 'cap_gpu']
SENSOR_COLS = ['job_name', 'task_name', 'worker_name', 'inst_id', 'machine', 'gpu_name', 'cpu_usage', 'gpu_wrk_util','avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count']

# Load data
print("Loading data...")
try:
    df_instance = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_instance_table/pai_instance_table.csv", header=None, names=INSTANCE_COLS)
    df_task = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_task_table/pai_task_table.csv", header=None, names=TASK_COLS)
    df_group = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_group_tag_table/pai_group_tag_table.csv", header=None, names=GROUP_COLS)
    df_job = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_job_table/pai_job_table.csv", header=None, names=JOB_COLS)
    df_machine_metric = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_machine_metric/pai_machine_metric.csv",header=None, names=MACHINE_METRIC_COLS)
    df_machine_spec = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_machine_spec/pai_machine_spec.csv",header=None, names=MACHINE_SPEC_COLS)
    df_sensor = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Dataset/pai_sensor_table/pai_sensor_table.csv",header=None, names=SENSOR_COLS)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the file paths are correct.")
    exit()


print(f"Original instance records: {len(df_instance):,}")
print(f"Original sensor records: {len(df_sensor):,}")
print(f"Original job records: {len(df_job):,}")
print(f"Original task records: {len(df_task):,}")
print(f"Original group records: {len(df_group):,}")
print(f"Original machine metric records: {len(df_machine_metric):,}")
print(f"Original machine metric spec: {len(df_machine_spec):}")

print("\nFiltering for Terminated status...")
df_instance = df_instance[df_instance['status'] == 'Terminated']
print(f"Terminated instances: {df_instance['status'].value_counts()}")

print("\nMerging datasets...")
df_merged = pd.merge(
    df_instance,
    df_sensor,
    on=['job_name', 'task_name', 'inst_id', 'worker_name', 'machine'],
    how='left',
    suffixes=('_instance', '_sensor'),
)
print(f"After instance+sensor merge: {len(df_merged):,} rows")
print(df_merged.columns)

# Calculate duration
df_merged['duration'] = (df_merged['end_time'] - df_merged['start_time'])

print("\nMerging with task data to get plan resources...")
df_merged = pd.merge(
    df_merged,
    df_task[['job_name', 'task_name', 'plan_cpu', 'plan_mem', 'plan_gpu', 'inst_number']],
    on=['job_name', 'task_name'],
    how='left'
)

print("\nMerging with job data to get submit_time and user...")
df_merged = pd.merge(
    df_merged,
    df_job[['job_name', 'inst_id', 'user', 'start_time']].rename(columns={'start_time': 'submit_time'}),
    on=['job_name', 'inst_id'],
    how='left'
)

print("\nMerging with group data...")
df_merged = pd.merge(
    df_merged,
    df_group,
    on=['inst_id', 'user'],
    how='left'
)

print("\nMerging with machine spec to get machine gpu_type...")
df_merged = pd.merge(
    df_merged,
    df_machine_spec[['machine', 'gpu_type', 'cap_cpu', 'cap_mem', 'cap_gpu']].rename(columns={'gpu_type': 'gpu_type_machine_spec'}),
    on=['machine'],
    how='left'
)

print("\nMerging with machine metric to get machine metrics...")
df_merged = pd.merge(
    df_merged,
    df_machine_metric[['worker_name', 'machine', 'machine_cpu_iowait', 'machine_cpu_kernel', 'machine_cpu_usr', 'machine_gpu', 'machine_load_1', 'machine_net_receive', 'machine_num_worker', 'machine_cpu']],
        on=['machine', 'worker_name'],
    how='left'
)

print(df_merged.columns)

# Display columns with NaN values in the merged dataset
print("\nColumns with NaN values in the merged dataset:")
nan_counts = df_merged.isnull().sum()
nan_columns = nan_counts[nan_counts > 0]
if nan_columns.empty:
    print("No columns with NaN values.")
else:
    print(nan_columns)

# Save the merged dataset
output_file_merged = "/mnt/newdisk/jess/Alibaba2020/Preprocess/instance_merged.csv"
df_merged.to_csv(output_file_merged, index=False)
print(f"\nSaved filtered dataset to {output_file_merged}")
