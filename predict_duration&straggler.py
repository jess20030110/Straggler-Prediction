import pandas as pd
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from cuml.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import cudf
import matplotlib. pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def print_step(message):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Track total execution time
total_start = time.time()

# 0. GPU Setup
print_step("0. Initializing GPU...")
GPU_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda" if GPU_AVAILABLE else "cpu")
print(f"{'⚡ GPU: ' + torch.cuda.get_device_name(0) + ' detected!' if GPU_AVAILABLE else '⚠️ GPU not found - Falling back to CPU'}")

# Define the MLP model
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_scale=8):
        super(MLPRegressor, self).__init__()
        
        # Calculate hidden dimensions with scale
        hidden_dim = input_dim * hidden_scale
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(), #activation function
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights for better training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x).squeeze()


# 1. Data Loading
print_step("1. Loading data...")
df = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Preprocess/filtered_grouped_data_new.csv")
print(f"Loaded {len(df):,} rows")
print("Columns:", df.columns.tolist())

# Convert submit_time to datetime if not already
df['submit_time'] = pd.to_datetime(df['submit_time'], unit='s') 
# # Verify the conversion
# print(df['submit_time'].head())

# Time since reference point (helps with trend detection)
reference_date = df['submit_time'].min()
df['days_since_start'] = (df['submit_time'] - reference_date).dt.days

# Extract multiple temporal components
df['submit_hour'] = df['submit_time'].dt.hour  # 0-23
df['submit_dayofweek'] = df['submit_time'].dt.dayofweek  # 0-6
df['submit_dayofyear'] = df['submit_time'].dt.dayofyear  # 1-365
df['submit_weekofyear'] = df['submit_time'].dt.isocalendar().week  # 1-52
df['submit_month'] = df['submit_time'].dt.month  # 1-12

# Cyclical encoding for periodic features
def cyclical_encode(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df.drop(col, axis=1)

for col, max_val in [('submit_hour', 24), 
                    ('submit_dayofweek', 7),
                    ('submit_dayofyear', 365),
                    ('submit_weekofyear', 52),
                    ('submit_month', 12)]:
    df = cyclical_encode(df, col, max_val)

# # Get unique values of the column 'gpu_type_machine_spec'
# unique_values = df['gpu_type_machine_spec'].unique()
# # Print the unique values
# print(unique_values)
# # Count how many times each unique value appears
# value_counts = df['gpu_type_machine_spec'].value_counts()
# # print("Frequency of each unique value:")
# # print(value_counts)

# Handle missing GPU data
drop_cols = [
    'machine_cpu', 'machine_cpu_iowait', 'workload', 
    'machine_num_worker', 'machine_cpu_kernel', 'gpu_name',
    'machine_load_1', 'machine_gpu', 'machine_cpu_usr',
    'machine_net_receive', 'gpu_type', 'gpu_type_spec'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

moderate_nan_cols = [
    'cpu_usage', 'avg_mem', 'max_mem',
    'read_count', 'write_count', 'read', 'write',
]

gpu_nan_cols = [
    'plan_gpu','avg_gpu_wrk_mem', 
    'max_gpu_wrk_mem', 'gpu_wrk_util',
]

# # Fill with 0 if NaN indicates "no usage" (e.g., GPU metrics for CPU-only tasks)
df[gpu_nan_cols] = df[gpu_nan_cols].fillna(0)
print(f"Number of rows after fill 0: {len(df)}")
# Drop rows with NaN in the specified columns
df = df.dropna(subset=moderate_nan_cols)
# Verify the result
print(f"Number of rows after dropping NaN: {len(df)}")
# print(df.columns)


# 2. Feature Engineering
print_step("2. Preparing features...")
# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['gpu_type_machine_spec'], prefix='gpu_type')

# *** FIXED: Split data before target encoding to prevent data leakage ***
print_step("3. Preparing data...")
# Define features and target first
temporal_features = ['submit_hour_sin', 'submit_hour_cos',
                     'submit_dayofweek_sin', 'submit_dayofweek_cos',
                     'submit_dayofyear_sin', 'submit_dayofyear_cos',
                     'days_since_start'
                    ]

base_features = ['plan_gpu', 'plan_mem', 'plan_cpu', 'cap_cpu',
            'cap_gpu', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
            'wait_time', 'inst_number', 'cap_mem', 'avg_gpu_wrk_mem', 
            'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count'] + \
           [col for col in df_encoded.columns if col.startswith('gpu_type')] + \
           temporal_features

# Remove any features that don't exist in the dataframe
base_features = [f for f in base_features if f in df_encoded.columns]
target = "duration"

# Do the train-test split BEFORE target encoding to prevent leakage
X_temp = df_encoded[base_features + ['user', 'group']].copy()
y = df_encoded[target].astype('float32')

# Create a proper 3-way split: train, validation, test
X_train_temp, X_temp_val_test, y_train, y_temp_val_test = train_test_split(
    X_temp, y, test_size=0.3, random_state=42
)

X_val_temp, X_test_temp, y_val, y_test = train_test_split(
    X_temp_val_test, y_temp_val_test, test_size=0.5, random_state=42
)

# Now perform target encoding ONLY on the training data
# Mean target encoding function for high cardinality features
def mean_target_encode(train_data, val_data, test_data, cat_column, target_column, alpha=5):
    global_mean = float(train_data[target_column].mean())
    agg = train_data.groupby(cat_column)[target_column].agg(['mean', 'count'])
    agg['encoded'] = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
    encoding_map = agg['encoded'].to_dict()
    
    # Apply encoding to all datasets
    train_encoded = train_data[cat_column].map(encoding_map).fillna(global_mean)
    val_encoded = val_data[cat_column].map(encoding_map).fillna(global_mean)
    test_encoded = test_data[cat_column].map(encoding_map).fillna(global_mean)
    
    return train_encoded, val_encoded, test_encoded

# Prepare target values for encoding
train_with_target = X_train_temp.copy()
train_with_target[target] = y_train
val_with_target = X_val_temp.copy()
val_with_target[target] = y_val
test_with_target = X_test_temp.copy()
test_with_target[target] = y_test

# Target encoding for high-cardinality variables
print("Target encoding user categories...")
user_train_encoded, user_val_encoded, user_test_encoded = mean_target_encode(
    train_with_target, val_with_target, test_with_target, 'user', 'duration'
)

print("Target encoding group categories...")
group_train_encoded, group_val_encoded, group_test_encoded = mean_target_encode(
    train_with_target, val_with_target, test_with_target, 'group', 'duration'
)

# Add encoded features to datasets
X_train_temp['user_encoded'] = user_train_encoded
X_train_temp['group_encoded'] = group_train_encoded

X_val_temp['user_encoded'] = user_val_encoded
X_val_temp['group_encoded'] = group_val_encoded

X_test_temp['user_encoded'] = user_test_encoded
X_test_temp['group_encoded'] = group_test_encoded

# Add resource utilization ratio features (requested vs. used)
print("Adding resource utilization ratio features...")
for dataset in [X_train_temp, X_val_temp, X_test_temp]:
    dataset['cpu_usage_ratio'] = dataset['cpu_usage'] / dataset['plan_cpu'].replace(0, 1)
    dataset['mem_usage_ratio'] = dataset['avg_mem'] / dataset['plan_mem'].replace(0, 1)
    if 'plan_gpu' in dataset.columns and 'gpu_wrk_util' in dataset.columns:
        dataset['gpu_usage_ratio'] = dataset['gpu_wrk_util'] / dataset['plan_gpu'].replace(0, 1)
    # Add wait time to execution time ratio - Note: we don't have duration in X datasets
    # So we'll calculate this later
# Final features to use
features = base_features + ['user_encoded', 'group_encoded', 
                           'cpu_usage_ratio', 'mem_usage_ratio']

if 'gpu_usage_ratio' in X_train_temp.columns:
    features.append('gpu_usage_ratio')
print(features)
# Prepare final datasets
X_train = X_train_temp[features].fillna(0).astype('float32')
X_val = X_val_temp[features].fillna(0).astype('float32')
X_test = X_test_temp[features].fillna(0).astype('float32')

# print("\n=== Feature Summary ===")
# print(f"Total features: {len(features)}")
# print("Numeric features stats:")
# print(X_train.describe())

# 5. Normalize features with StandardScaler
print_step("5. Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Use quantile transformation for heavy-tailed duration data
target_scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
y_train_transformed = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_transformed = target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_transformed = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_transformed)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val_transformed)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_transformed)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 6. Model Training
print_step("6. Training model...")
train_start = time.time()

# Initialize model
input_dim = X_train_scaled.shape[1]
model = MLPRegressor(input_dim, hidden_scale=10).to(device)
print(f"Model architecture: {model}")

# Define loss and optimizer
# criterion = nn.MSELoss()  # Using MSELoss for regression
# criterion = nn.L1Loss()  # MAE Loss
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# # Learning rate scheduler
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# # Early stopping parameters
# best_loss = float('inf')
# patience = 10
# counter = 0
# early_stop = False

# # Training loop
# num_epochs = 100
# print("\nTraining progress:")
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
    
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         # Zero the parameter gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
        
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item() * inputs.size(0)
    
#     epoch_loss = running_loss / len(train_dataset)
    
#     # Validation - now uses validation set instead of test set
#     model.eval()
#     with torch.no_grad():
#         X_val_tensor_device = X_val_tensor.to(device)
#         y_val_tensor_device = y_val_tensor.to(device)
#         y_pred = model(X_val_tensor_device)
#         val_loss = criterion(y_pred, y_val_tensor_device).item()
    
#     # Update learning rate
#     scheduler.step(val_loss)

#     # Print statistics every 5 epochs
#     if (epoch + 1) % 5 == 0:
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
    
#     # Sample predictions every 10 epochs
#     if epoch % 10 == 0:
#         with torch.no_grad():
#             # Sample predictions using validation data
#             sample_idx = torch.randint(0, len(X_val_tensor_device), (5,))
#             sample_pred = model(X_val_tensor_device[sample_idx])
#             sample_true = y_val_tensor_device[sample_idx]
            
#             print("\nSample Predictions:")
#             for i in range(5):
#                 pred = target_scaler.inverse_transform(sample_pred[i].cpu().numpy().reshape(1,-1))[0][0]
#                 true = target_scaler.inverse_transform(sample_true[i].cpu().numpy().reshape(1,-1))[0][0]
#                 print(f"Pred: {pred:.1f}s | True: {true:.1f}s | Error: {abs(pred-true):.1f}s")
                
#     # Early stopping
#     if val_loss < best_loss:
#         best_loss = val_loss
#         counter = 0
#         # Save the best model
#         torch.save(model.state_dict(), '/mnt/newdisk/jess/2020_Trace/Result/best_mlp_model_MAE_modified.pth')
#     else:
#         counter += 1
#         if counter >= patience:
#             print(f'Early stopping at epoch {epoch+1}')
#             early_stop = True
#             break

# Load best model
model.load_state_dict(torch.load('/mnt/newdisk/jess/2020_Trace/Result/best_mlp_model_MAE_modified.pth'))
print(f"Model trained in {time.time() - train_start:.2f} seconds")

# 7. Evaluation
print_step("7. Evaluating model...")
model.eval()
with torch.no_grad():
    # Move test data to device
    X_test_gpu = X_test_tensor.to(device)
    
    # Generate predictions
    predictions = model(X_test_gpu).cpu().numpy()
    
    # Transform back to original scale
    y_test_original = target_scaler.inverse_transform(y_test_transformed.reshape(-1, 1)).flatten()
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Calculate various metrics on original scale
mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
r2 = r2_score(y_test_original, predictions_original)

print(f"Model MAE: {mae:.2f} seconds")
print(f"Model RMSE: {rmse:.2f} seconds")
print(f"R² Score: {r2:.4f}")

# Calculate percentage error with proper handling of zeros
valid_mask = y_test_original != 0
if np.any(valid_mask):
    # Calculate absolute percentage error for non-zero values
    ape = np.abs((y_test_original[valid_mask] - predictions_original[valid_mask]) / 
                  y_test_original[valid_mask]) * 100
    mape = np.mean(ape)
    
    # Calculate median absolute percentage error (more robust to outliers)
    mdape = np.median(ape)
    
    print(f"Mean Absolute Percentage Error (non-zero): {mape:.2f}%")
    print(f"Median Absolute Percentage Error (non-zero): {mdape:.2f}%")
    
    # Calculate percentage of predictions within 20% of actual
    within_20pct = np.mean(ape <= 20) * 100
    print(f"Predictions within 20% of actual: {within_20pct:.2f}%")
else:
    print("Warning: All true durations are zero - cannot calculate percentage error")

# 8. Generate predictions for all data
print_step("8. Preparing output...")

# Now we need to prepare the entire dataset for prediction
# We'll reuse the target encoding from before
df_all = df_encoded.copy()

# Apply encoding to all data (using the same encoding as for train/val/test)
# For simplicity, we'll recreate the encoding maps
train_with_target = X_train_temp.copy()
train_with_target[target] = y_train

encoding_maps = {}
for col in ['user', 'group']:
    global_mean = float(train_with_target[target].mean())
    agg = train_with_target.groupby(col)[target].agg(['mean', 'count'])
    agg['encoded'] = (agg['count'] * agg['mean'] + 5 * global_mean) / (agg['count'] + 5)
    encoding_maps[col] = agg['encoded'].to_dict()

# Apply encodings to all data
df_all['user_encoded'] = df_all['user'].map(encoding_maps['user']).fillna(global_mean)
df_all['group_encoded'] = df_all['group'].map(encoding_maps['group']).fillna(global_mean)

# Add derived features
df_all['cpu_usage_ratio'] = df_all['cpu_usage'] / df_all['plan_cpu'].replace(0, 1)
df_all['mem_usage_ratio'] = df_all['avg_mem'] / df_all['plan_mem'].replace(0, 1)
if 'plan_gpu' in df_all.columns and 'gpu_wrk_util' in df_all.columns:
    df_all['gpu_usage_ratio'] = df_all['gpu_wrk_util'] / df_all['plan_gpu'].replace(0, 1)

# Get all features
X_all = df_all[features].fillna(0).astype('float32')

# Scale all data
X_all_scaled = scaler.transform(X_all)
X_all_tensor = torch.FloatTensor(X_all_scaled).to(device)

# Generate predictions
with torch.no_grad():
    all_predictions = model(X_all_tensor).cpu().numpy()
    all_predictions = target_scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()

# Add predictions to the dataframe
df_all['predicted_duration'] = all_predictions

# Calculate absolute and percentage errors
df_all['absolute_error'] = np.abs(df_all['duration'] - df_all['predicted_duration'])

# Calculate percentage error with proper handling of zeros
df_all['absolute_percentage_error'] = np.where(
    df_all['duration'] != 0,
    np.abs((df_all['duration'] - df_all['predicted_duration']) / df_all['duration']) * 100,
    np.nan  # Use NaN instead of 0 for zero durations
)

# =====================================================================
# IMPROVED KNN STRAGGLER DETECTION
# =====================================================================
print_step("9. KNN Straggler Detection Implementation...")

# 9.1 Define ground truth stragglers
print_step("9.1 Define ground truth stragglers...")
# Group by job_name and task_name to define what constitutes a straggler
thresholds = df_all.groupby(['job_name', 'task_name'])['duration'].quantile(0.90).reset_index()
thresholds = thresholds.rename(columns={'duration': 'duration_threshold'})
df_all = df_all.merge(thresholds, on=['job_name', 'task_name'], how='left')
df_all['is_straggler'] = (df_all['duration'] > df_all['duration_threshold']).astype(int)

# Count stragglers vs non-stragglers
straggler_count = df_all['is_straggler'].sum()
print(f"Ground truth stragglers: {straggler_count} ({straggler_count/len(df_all)*100:.2f}% of tasks)")

# Feature Engineering - Keep only the most discriminative features
df_all['duration_ratio'] = df_all['duration'] / df_all['predicted_duration'].clip(1e-6, None)
df_all['duration_diff'] = df_all['duration'] - df_all['predicted_duration']
df_all['resource_adequacy'] = (df_all['plan_gpu'] * df_all['plan_cpu']) / (df_all['duration'] + 1e-6)
df_all['wait_ratio'] = df_all['wait_time'] / (df_all['duration'] + 1e-6)
df_all['prediction_error'] = (df_all['duration'] - df_all['predicted_duration']).abs()
df_all['relative_error'] = df_all['prediction_error'] / (df_all['duration'] + 1e-6)

# Clipping to avoid extreme values
df_all['duration_ratio'] = df_all['duration_ratio'].clip(0, 10)  # More reasonable clipping
df_all['resource_adequacy'] = df_all['resource_adequacy'].clip(0, 1000)
df_all['wait_ratio'] = df_all['wait_ratio'].clip(0, 10)
df_all['relative_error'] = df_all['relative_error'].clip(0, 5)

# FEATURE SELECTION: Use a more discriminative but smaller feature set for KNN
# This helps avoid the curse of dimensionality
X = df_all[[
    'predicted_duration',     # MLP prediction (strong predictor)
    'duration_ratio',         # How much longer than expected (key indicator)
    'wait_time',              # Wait time before execution
    'plan_gpu', 'plan_cpu',   # Resource requests (core metrics)
    'cpu_usage', 'gpu_wrk_util',  # Actual resource utilization 
    'wait_ratio',             # Wait time ratio (potentially important)
    'resource_adequacy',      # Resource planning metric
    'prediction_error',       # Absolute prediction error
    'days_since_start',       # Time factor
]]
y = df_all['is_straggler']

X_pd = X
y_pd = y

# Create stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pd, y_pd, test_size=0.2, random_state=42, stratify=y_pd
)
# Print counts before SMOTE (in training set)
print("Before SMOTE:")
print("Non-stragglers (0):", (y_train == 0).sum())
print("Stragglers (1):", (y_train == 1).sum())

# Print counts in test set (unchanged)
print("\nTest Set Distribution:")
print("Non-stragglers (0):", (y_test == 0).sum())
print("Stragglers (1):", (y_test == 1).sum())

# Save the original test set for later evaluation
X_test_original = X_test.copy()
y_test_original = y_test.copy()

# === Scale first, then apply SMOTE ===
print("Applying scaling and SMOTE...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
# Print counts after SMOTE (in training set)
print("\nAfter SMOTE:")
y_train_resampled_series = pd.Series(y_train_resampled)
print("Non-stragglers (0):", (y_train_resampled_series == 0).sum())
print("Stragglers (1):", (y_train_resampled_series == 1).sum())

# Convert to float32 for efficiency
X_train_scaled = X_train_resampled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)

# Convert to GPU if available
if GPU_AVAILABLE:
    X_train_scaled = cudf.DataFrame(X_train_scaled)
    X_test_scaled = cudf.DataFrame(X_test_scaled)
    y_train_resampled = cudf.Series(y_train_resampled)
    y_test = cudf.Series(y_test.values)

print("KNN feature preparation complete!")

# 9.3 Train and evaluate KNN model with hyperparameter tuning
print_step("9.2 Training KNN models with hyperparameter tuning...")

# Define function to convert GPU tensors to numpy if needed
def convert_to_numpy(data):
    if GPU_AVAILABLE:
        if hasattr(data, 'to_numpy'):
            return data.to_numpy()
        elif hasattr(data, 'values'):
            return data.values
        elif hasattr(data, 'get'):
            return data.get()
    return data

# Hyperparameter tuning with grid search
y_train_np = convert_to_numpy(y_train_resampled)
y_test_np = convert_to_numpy(y_test)

best_f1 = 0
best_k = 0
best_metric = ''
best_threshold = 0
best_model = None
best_predictions = None
best_probas = None

# Try different K values and distance metrics (more comprehensive)
k_values = [3, 5, 7, 9, 11, 15]
metrics = ['euclidean', 'manhattan', 'chebyshev']

print("Starting grid search for KNN parameters...")
results = []

for k in k_values:
    for metric in metrics:
        print(f"Testing k={k}, metric={metric}")
        
        # Initialize and train KNN
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train_scaled, y_train_np)
        
        # Predict probabilities
        y_probs = knn.predict_proba(X_test_scaled)
        y_probs_np = convert_to_numpy(y_probs)[:, 1]
        
        # Find optimal threshold using F1 score
        precision, recall, thresholds = precision_recall_curve(y_test_np, y_probs_np)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        if len(thresholds) > 0:  # Handle edge case
            optimal_idx = np.argmax(f1_scores[:-1])  # Exclude the last item which doesn't have a threshold
            optimal_threshold = thresholds[optimal_idx]
            
            # Make predictions with optimal threshold
            y_pred = (y_probs_np >= optimal_threshold).astype(int)
            
            # Calculate metrics
            f1 = f1_score(y_test_np, y_pred)
            precision_val = precision_score(y_test_np, y_pred)
            recall_val = recall_score(y_test_np, y_pred)
            accuracy = accuracy_score(y_test_np, y_pred)
            roc_auc = roc_auc_score(y_test_np, y_probs_np)
            
            # Store results
            results.append({
                'k': k,
                'metric': metric,
                'threshold': optimal_threshold,
                'f1': f1,
                'precision': precision_val,
                'recall': recall_val,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            })
            
            print(f"  F1={f1:.4f}, Precision={precision_val:.4f}, Recall={recall_val:.4f}")
            
            # Update best model if this one is better
            if f1 > best_f1:
                best_f1 = f1
                best_k = k
                best_metric = metric
                best_threshold = optimal_threshold
                best_model = knn
                best_predictions = y_pred
                best_probas = y_probs_np

# Display grid search results as a sorted table
results_df = pd.DataFrame(results)
print("\nGrid Search Results (sorted by F1 score):")
print(results_df.sort_values('f1', ascending=False).head(5))

# === Evaluation of best model ===
print_step("10. Evaluating Best KNN Model")
print(f"\nBest model: k={best_k}, metric={best_metric}, threshold={best_threshold:.3f}")

# Calculate detailed metrics for best model
tn, fp, fn, tp = confusion_matrix(y_test_np, best_predictions).ravel()

print(f"\nKNN Performance (Best Model):")
print(f"Confusion Matrix:\n[[{tn:5} {fp:5}]\n [{fn:5} {tp:5}]]")
print(f"F1 Score: {best_f1:.4f}")
print(f"Precision: {precision_score(y_test_np, best_predictions):.4f}")
print(f"Recall: {recall_score(y_test_np, best_predictions):.4f}")
print(f"Accuracy: {accuracy_score(y_test_np, best_predictions):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test_np, best_probas):.4f}")
print(f"Avg Precision: {average_precision_score(y_test_np, best_probas):.4f}")

# Feature importance using permutation importance
print_step("11. Feature importance analysis...")
try:
    # Run permutation importance on the best model
    perm_importance = permutation_importance(
        best_model, X_test_scaled, y_test_np, 
        n_repeats=10, random_state=42
    )
    
    # Create a DataFrame with feature importances
    feature_names = X.columns.tolist()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance (top 10):")
    print(importance_df.head(10))
    
    # Save feature importance
    importance_df.to_csv('feature_importance_knn.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1])
    plt.xlabel('Mean Decrease in Accuracy')
    plt.title('Top 10 Feature Importance (KNN Straggler Detection)')
    plt.tight_layout()
    plt.savefig('knn_feature_importance.png')
    plt.close()
    
except Exception as e:
    print(f"Could not calculate feature importance: {e}")

# Save the best model predictions
try:
    # Add predictions to the original DataFrame
    df_all['knn_straggler_probability'] = np.nan
    df_all.loc[X_test_original.index, 'knn_straggler_probability'] = best_probas
    df_all['knn_straggler_prediction'] = np.nan
    df_all.loc[X_test_original.index, 'knn_straggler_prediction'] = best_predictions
    
    # Calculate additional metrics for model analysis
    df_all['straggler_true_positive'] = ((df_all['is_straggler'] == 1) & 
                                         (df_all['knn_straggler_prediction'] == 1)).astype(int)
    df_all['straggler_false_negative'] = ((df_all['is_straggler'] == 1) & 
                                          (df_all['knn_straggler_prediction'] == 0)).astype(int)
    
    # Save key columns for analysis
    straggler_results = df_all[[
        'inst_id', 'job_name', 'task_name', 'duration', 'predicted_duration',
        'is_straggler', 'knn_straggler_probability', 'knn_straggler_prediction',
        'straggler_true_positive', 'straggler_false_negative',
        'duration_ratio', 'wait_time', 'resource_adequacy'
    ]].copy()
    
    # Save to CSV
    straggler_results.to_csv('knn_straggler_predictions.csv', index=False)
    print("Straggler predictions saved to 'knn_straggler_predictions.csv'")
    
    # Create visualization of model performance
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test_np, best_probas)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test_np, best_probas):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for KNN Straggler Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('knn_straggler_roc_curve.png')
    plt.close()
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test_np, best_probas)
    plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision_score(y_test_np, best_probas):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for KNN Straggler Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('knn_straggler_pr_curve.png')
    
except Exception as e:
    print(f"Error in saving predictions: {e}")

print("\nKNN straggler detection completed!")
