import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
from datetime import datetime
from sklearn.preprocessing import TargetEncoder, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def print_step(message):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Track total execution time
total_start = time.time()

# 1. Data Loading
print_step("1. Loading data...")
load_start = time.time()
df = pd.read_csv("filtered_grouped_data.csv")
load_time = time.time() - load_start
print(f"Loaded {len(df):,} rows in {load_time:.2f} seconds")
print("Columns:", df.columns.tolist())

# Handle missing GPU data
df['plan_gpu'] = df['plan_gpu'].fillna(0) 
df['gpu_type'] = df['gpu_type'].fillna('NO_GPU')  # Combine with gpu_type_spec
if 'gpu_type_spec' in df.columns:
    df['gpu_type'] = df['gpu_type'].combine_first(df['gpu_type_spec'])
    df = df.drop(columns=['gpu_type_spec'])

load_time = time.time() - load_start
print(f"Loaded {len(df):,} rows in {load_time:.2f} seconds")
print("Columns:", df.columns.tolist())

# Modified encoding approach
print_step("2. Preparing features...")
prep_start = time.time()

# 1. One-hot encode combined GPU types
df_encoded = pd.get_dummies(df, columns=['gpu_type'], prefix='gpu_')


# 2. Frequency encode users
user_counts = df['user'].value_counts()
df_encoded['user_count'] = df['user'].map(user_counts)

# 3. Target encode groups
# encoder = TargetEncoder()
# df_encoded['group_encoded'] = encoder.fit_transform(df[['group']], df['duration'])
# df_encoded['group_encoded'] = MinMaxScaler().fit_transform(df_encoded[['group_encoded']])
df_encoded['group_encoded'] = LabelEncoder().fit_transform(df['group'])

# Show transformed data
print("\nAfter Encoding:")
print(df_encoded)
print("\nEncoded Data Types:")
print(df_encoded.dtypes)

# Final features
features = ['user_count', 'plan_gpu', 'plan_mem', 'plan_cpu'] + \
           [col for col in df_encoded.columns if col.startswith('gpu_')] + \
           ['group_encoded']
target = "duration"
print(features)
prep_time = time.time() - prep_start
print(f"Prepared {len(features)} features in {prep_time:.2f} seconds") 


# 3. Train-Test Split
print_step("3. Splitting data...")
split_start = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded[features], 
    df[target], 
    test_size=0.2, 
    random_state=42
)
split_time = time.time() - split_start
print(f"Split into {len(X_train):,} train and {len(X_test):,} test samples in {split_time:.2f} seconds")

# 4. Model Training
print_step("4. Training model...")
train_start = time.time()
model = DecisionTreeRegressor(
    max_leaf_nodes=10,
    criterion='absolute_error',
    random_state=42
)
model.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"Trained model in {train_time:.2f} seconds")

# 5. Evaluation
print_step("5. Evaluating model...")
eval_start = time.time()
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
eval_time = time.time() - eval_start
print(f"Model MAE: {mae:.2f} seconds (calculated in {eval_time:.2f}s)")

# 6. Generating Output
print_step("6. Preparing output...")
output_start = time.time()
df['predicted_duration'] = model.predict(df_encoded[features])

output_columns = [
    'inst_id', 'user', 'plan_gpu', 'plan_cpu', 'plan_mem', 
    'inst_number', 'duration', 'predicted_duration', 'job_name', 
    'submit_time', 'group', 'gpu_type', 'wait_time'
]

output_df = df[output_columns].rename(columns={'inst_id': 'job_id'})
output_file = "cart_predictions.csv"
output_df.to_csv(output_file, index=False)
output_time = time.time() - output_start
print(f"Saved predictions to {output_file} in {output_time:.2f} seconds")

# Final summary
total_time = time.time() - total_start
print_step(f"Completed! Total execution time: {total_time:.2f} seconds")
print("="*50)
print("Final output columns:", output_df.columns.tolist())
print(f"First prediction sample:\n{output_df.iloc[0]}")