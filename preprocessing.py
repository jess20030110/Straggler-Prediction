import pandas as pd

# Define column names
TASK_COLS = ['job_name', 'task_name', 'inst_number', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type']
GROUP_COLS = ['inst_id', 'user', 'gpu_type_spec', 'group', 'workload']
JOB_COLS = ['job_name', 'inst_id', 'user', 'status', 'start_time', 'end_time']
INSTANCE_COLS = ['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id', 'status', 'start_time', 'end_time', 'machine']

# Load data
print("Loading data...")
df_instance = pd.read_csv("/mnt/newdisk/jess/2020_Trace/pai_instance_table/pai_instance_table.csv", header=None, names=INSTANCE_COLS)
df_task = pd.read_csv("/mnt/newdisk/jess/2020_Trace/pai_task_table/pai_task_table.csv", header=None, names=TASK_COLS)
df_group = pd.read_csv("/mnt/newdisk/jess/2020_Trace/pai_group_tag_table/pai_group_tag_table.csv", header=None, names=GROUP_COLS)
df_job = pd.read_csv("/mnt/newdisk/jess/2020_Trace/pai_job_table/pai_job_table.csv", header=None, names=JOB_COLS)
print(f"Instance: {len(df_instance):,} | Task: {len(df_task):,} | Job: {len(df_job):,}")

# 1. Filter for Terminated status
print("\nFiltering for Terminated status...")
df_instance = df_instance[df_instance['status'] == 'Terminated']
df_task = df_task[df_task['status'] == 'Terminated']
df_job = df_job[df_job['status'] == 'Terminated']
print(f"Instance: {len(df_instance):,} | Task: {len(df_task):,} | Job: {len(df_job):,}")

# 2. Merge four datasets
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

# Final merge: (task + job + instance) + group (on inst_id)
df_final = pd.merge(
    df_merged,
    df_group,
    on=['inst_id', 'user'],  # Now inst_id is consistent
    suffixes=('', '_group')
)
print(f"\nTotal rows after merging group: {len(df_final):,}")

# 3. Filtering steps
print("\nApplying group filters...")

# 3.1 Count unique groups
unique_groups = df_final['group'].nunique()
print(f"Number of unique groups: {unique_groups:,}")

# 3.2 Count group frequencies
group_counts = df_final['group'].value_counts().reset_index()
group_counts.columns = ['group', 'count']
print("\nTop 10 groups by frequency:")
print(group_counts.head(10))

# 3.3 Filter for groups appearing at least 5 times
min_group_count = 5
frequent_groups = group_counts[group_counts['count'] >= min_group_count]['group']
dfa_filtered = df_final[df_final['group'].isin(frequent_groups)]

print(f"\nRows after keeping only groups with ≥{min_group_count} occurrences: {len(dfa_filtered):,}")
print(f"Number of groups remaining: {dfa_filtered['group'].nunique():,}")

# Final output
print("\nFinal filtered data:")
print(f"Columns with merging: {df_final.columns}")
print(dfa_filtered.head())

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

# Final output
print("\nFinal filtered data with new columns:")
print(f"Columns: {dfa_filtered.columns}")
print(dfa_filtered.head())

# 5. Save the filtered and grouped (but not aggregated) data
output_file = "filtered_grouped_data.csv"
dfa_filtered.to_csv(output_file, index=False)
print(f"\nSaved the complete filtered and grouped (but not aggregated) data to {output_file}")