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
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.pipeline import Pipeline

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
            nn.ReLU(),
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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x).squeeze()

# 1. Data Loading and Preprocessing
print_step("1. Loading and preprocessing data...")
df = pd.read_csv("/mnt/newdisk/jess/2020_Trace/Preprocess/filtered_grouped_data_new.csv")
print(f"Loaded {len(df):,} rows")

# Convert submit_time to datetime
df['submit_time'] = pd.to_datetime(df['submit_time'], unit='s') 

# Time-based feature engineering
reference_date = df['submit_time'].min()
df['days_since_start'] = (df['submit_time'] - reference_date).dt.days
df['submit_hour'] = df['submit_time'].dt.hour
df['submit_dayofweek'] = df['submit_time'].dt.dayofweek
df['submit_dayofyear'] = df['submit_time'].dt.dayofyear
df['submit_weekofyear'] = df['submit_time'].dt.isocalendar().week
df['submit_month'] = df['submit_time'].dt.month

# Cyclical encoding for periodic features
def cyclical_encode(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df.drop(col, axis=1)

for col, max_val in [('submit_hour', 24), ('submit_dayofweek', 7),
                     ('submit_dayofyear', 365), ('submit_weekofyear', 52),
                     ('submit_month', 12)]:
    df = cyclical_encode(df, col, max_val)

# Handle missing data
drop_cols = ['machine_cpu', 'machine_cpu_iowait', 'workload', 
             'machine_num_worker', 'machine_cpu_kernel', 'gpu_name',
             'machine_load_1', 'machine_gpu', 'machine_cpu_usr',
             'machine_net_receive', 'gpu_type', 'gpu_type_spec']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

moderate_nan_cols = ['cpu_usage', 'avg_mem', 'max_mem',
                     'read_count', 'write_count', 'read', 'write']
gpu_nan_cols = ['plan_gpu','avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'gpu_wrk_util']

df[gpu_nan_cols] = df[gpu_nan_cols].fillna(0)
df = df.dropna(subset=moderate_nan_cols)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['gpu_type_machine_spec'], prefix='gpu_type')

# Reset index for consistency
df = df.reset_index(drop=True)
print(f"Final dataset shape: {df.shape}")

# 2. Define features and target
temporal_features = ['submit_hour_sin', 'submit_hour_cos',
                     'submit_dayofweek_sin', 'submit_dayofweek_cos',
                     'submit_dayofyear_sin', 'submit_dayofyear_cos',
                     'days_since_start']

base_features = ['plan_gpu', 'plan_mem', 'plan_cpu', 'cap_cpu',
                 'cap_gpu', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
                 'wait_time', 'inst_number', 'cap_mem', 'avg_gpu_wrk_mem', 
                 'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count'] + \
                [col for col in df.columns if col.startswith('gpu_type')] + \
                temporal_features

base_features = [f for f in base_features if f in df.columns]
target = "duration"

# 3. FIXED: Proper train/validation/test split
print_step("3. Creating train/validation/test splits...")
X_temp = df[base_features + ['user', 'group']].copy()
y = df[target].astype('float32')

# Split into train (60%), validation (20%), test (20%)
X_train_temp, X_temp_remaining, y_train, y_temp_remaining = train_test_split(
    X_temp, y, test_size=0.4, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
)

X_val_temp, X_test_temp, y_val, y_test = train_test_split(
    X_temp_remaining, y_temp_remaining, test_size=0.5, random_state=42, 
    stratify=pd.qcut(y_temp_remaining, q=5, duplicates='drop')
)

print(f"Train size: {len(X_train_temp)} ({len(X_train_temp)/len(df)*100:.1f}%)")
print(f"Validation size: {len(X_val_temp)} ({len(X_val_temp)/len(df)*100:.1f}%)")
print(f"Test size: {len(X_test_temp)} ({len(X_test_temp)/len(df)*100:.1f}%)")

# 4. FIXED: Target encoding using only training data
print_step("4. Target encoding (no data leakage)...")

def mean_target_encode(train_data, val_data, test_data, cat_column, target_column, alpha=5):
    """Target encoding that prevents data leakage"""
    global_mean = float(train_data[target_column].mean())
    
    # Calculate encoding from training data only
    agg = train_data.groupby(cat_column)[target_column].agg(['mean', 'count'])
    agg['encoded'] = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
    encoding_map = agg['encoded'].to_dict()
    
    # Apply encoding to all datasets
    train_encoded = train_data[cat_column].map(encoding_map).fillna(global_mean)
    val_encoded = val_data[cat_column].map(encoding_map).fillna(global_mean)
    test_encoded = test_data[cat_column].map(encoding_map).fillna(global_mean)
    
    return train_encoded, val_encoded, test_encoded, encoding_map

# Prepare datasets with target for encoding
train_with_target = X_train_temp.copy()
train_with_target[target] = y_train
val_with_target = X_val_temp.copy()
val_with_target[target] = y_val
test_with_target = X_test_temp.copy()
test_with_target[target] = y_test

# Target encoding
user_train_enc, user_val_enc, user_test_enc, user_encoding_map = mean_target_encode(
    train_with_target, val_with_target, test_with_target, 'user', 'duration'
)

group_train_enc, group_val_enc, group_test_enc, group_encoding_map = mean_target_encode(
    train_with_target, val_with_target, test_with_target, 'group', 'duration'
)

# Add encoded features
for dataset, user_enc, group_enc in [(X_train_temp, user_train_enc, group_train_enc),
                                     (X_val_temp, user_val_enc, group_val_enc),
                                     (X_test_temp, user_test_enc, group_test_enc)]:
    dataset['user_encoded'] = user_enc
    dataset['group_encoded'] = group_enc
    
    # Add resource utilization ratios
    dataset['cpu_usage_ratio'] = dataset['cpu_usage'] / dataset['plan_cpu'].replace(0, 1)
    dataset['mem_usage_ratio'] = dataset['avg_mem'] / dataset['plan_mem'].replace(0, 1)
    dataset['gpu_usage_ratio'] = dataset['gpu_wrk_util'] / dataset['plan_gpu'].replace(0, 1)

# Final feature set
features = base_features + ['user_encoded', 'group_encoded', 
                           'cpu_usage_ratio', 'mem_usage_ratio', 'gpu_usage_ratio']

# Prepare final datasets
X_train = X_train_temp[features].fillna(0).astype('float32')
X_val = X_val_temp[features].fillna(0).astype('float32')
X_test = X_test_temp[features].fillna(0).astype('float32')

# 5. Feature scaling
print_step("5. Feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Target scaling
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

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 6. FIXED: Model Training with clear loss function
print_step("6. Training MLP model...")
train_start = time.time()

input_dim = X_train_scaled.shape[1]
model = MLPRegressor(input_dim, hidden_scale=10).to(device)

# FIXED: Use only one loss function consistently
criterion = nn.L1Loss()  # MAE Loss - clearly defined
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Training loop
best_val_loss = float('inf')
patience_counter = 0
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_val_device = X_val_tensor.to(device)
        y_val_device = y_val_tensor.to(device)
        y_pred = model(X_val_device)
        val_loss = criterion(y_pred, y_val_device).item()
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), '/mnt/newdisk/jess/2020_Trace/Result/best_mlp_model_fixed.pth')
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Load best model
model.load_state_dict(torch.load('/mnt/newdisk/jess/2020_Trace/Result/best_mlp_model_fixed.pth'))
print(f"Model trained in {time.time() - train_start:.2f} seconds")

# 7. Model Evaluation
print_step("7. Evaluating MLP model...")
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    predictions = model(X_test_device).cpu().numpy()
    
    # Transform back to original scale
    y_test_original = target_scaler.inverse_transform(y_test_transformed.reshape(-1, 1)).flatten()
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
r2 = r2_score(y_test_original, predictions_original)

print(f"MLP Model Performance:")
print(f"MAE: {mae:.2f} seconds")
print(f"RMSE: {rmse:.2f} seconds")
print(f"R² Score: {r2:.4f}")

# Calculate MAPE for non-zero values
valid_mask = y_test_original != 0
if np.any(valid_mask):
    ape = np.abs((y_test_original[valid_mask] - predictions_original[valid_mask]) / 
                  y_test_original[valid_mask]) * 100
    mape = np.mean(ape)
    print(f"MAPE (non-zero): {mape:.2f}%")

    # Calculate percentage of predictions within 20% of actual
    within_25pct = np.mean(ape <= 25) * 100
    print(f"Predictions within 25% of actual: {within_25pct:.2f}%")
else:
    print("Warning: All true durations are zero - cannot calculate percentage error")

# 8. Generate predictions for all data
print_step("8. Generating predictions for full dataset...")

# Apply same preprocessing to full dataset
df_full = df.copy()

# Apply target encodings using training-derived mappings
global_mean = float(train_with_target['duration'].mean())
df_full['user_encoded'] = df_full['user'].map(user_encoding_map).fillna(global_mean)
df_full['group_encoded'] = df_full['group'].map(group_encoding_map).fillna(global_mean)

# Add derived features
df_full['cpu_usage_ratio'] = df_full['cpu_usage'] / df_full['plan_cpu'].replace(0, 1)
df_full['mem_usage_ratio'] = df_full['avg_mem'] / df_full['plan_mem'].replace(0, 1)
df_full['gpu_usage_ratio'] = df_full['gpu_wrk_util'] / df_full['plan_gpu'].replace(0, 1)

X_full = df_full[features].fillna(0).astype('float32')
X_full_scaled = scaler.transform(X_full)
X_full_tensor = torch.FloatTensor(X_full_scaled).to(device)

# Generate predictions
with torch.no_grad():
    full_predictions = model(X_full_tensor).cpu().numpy()
    full_predictions = target_scaler.inverse_transform(full_predictions.reshape(-1, 1)).flatten()

df_full['predicted_duration'] = full_predictions
df_full['absolute_error'] = np.abs(df_full['duration'] - df_full['predicted_duration'])

# 9. FIXED: Straggler Detection (No Data Leakage)
print_step("9. Straggler detection (training data only)...")

# FIXED: Calculate straggler thresholds using only training data
train_indices = X_train_temp.index
train_data = df_full.loc[train_indices].copy()

# Calculate thresholds from training data only
train_thresholds = train_data.groupby(['job_name', 'task_name'])['duration'].quantile(0.90).reset_index()
train_thresholds = train_thresholds.rename(columns={'duration': 'duration_threshold'})

# Apply thresholds to full dataset
df_full = df_full.merge(train_thresholds, on=['job_name', 'task_name'], how='left')

# For job/task combinations not in training, use global threshold
global_threshold = train_data['duration'].quantile(0.90)
df_full['duration_threshold'] = df_full['duration_threshold'].fillna(global_threshold)
df_full['is_straggler'] = (df_full['duration'] > df_full['duration_threshold']).astype(int)

print(f"Stragglers: {df_full['is_straggler'].sum()} ({df_full['is_straggler'].mean()*100:.2f}%)")

# 10. FIXED: KNN Straggler Detection
print_step("10. KNN straggler detection...")

# Prepare KNN features
knn_features = [
    'predicted_duration', 'wait_time', 'plan_gpu', 'plan_cpu',
    'cpu_usage', 'gpu_wrk_util', 'days_since_start'
]

# Add derived features for KNN
df_full['duration_ratio'] = df_full['duration'] / df_full['predicted_duration'].clip(1e-6, None)
df_full['duration_ratio'] = df_full['duration_ratio'].clip(0, 10)  # Cap extreme values

knn_features.append('duration_ratio')

# Prepare KNN datasets using the same splits
X_train_knn = df_full.loc[train_indices, knn_features].values
y_train_knn = df_full.loc[train_indices, 'is_straggler'].values

X_val_knn = df_full.loc[X_val_temp.index, knn_features].values
y_val_knn = df_full.loc[X_val_temp.index, 'is_straggler'].values

X_test_knn = df_full.loc[X_test_temp.index, knn_features].values
y_test_knn = df_full.loc[X_test_temp.index, 'is_straggler'].values

print(f"KNN Training set: {X_train_knn.shape}, Stragglers: {y_train_knn.mean():.3f}")
print(f"KNN Test set: {X_test_knn.shape}, Stragglers: {y_test_knn.mean():.3f}")

# FIXED: Proper KNN hyperparameter tuning
best_f1 = 0
best_params = {}
best_model = None

k_values = [3, 5, 7, 9, 11]
metrics = ['euclidean', 'manhattan']

print("KNN Hyperparameter tuning:")
for k in k_values:
    for metric in metrics:
        # Use validation set for hyperparameter tuning
        knn_scaler = StandardScaler()
        X_train_knn_scaled = knn_scaler.fit_transform(X_train_knn)
        X_val_knn_scaled = knn_scaler.transform(X_val_knn)
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_knn_scaled, y_train_knn)
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train_balanced, y_train_balanced)
        
        # Predict on validation set
        y_val_proba = knn.predict_proba(X_val_knn_scaled)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_val_knn, y_val_proba)
        if len(thresholds) > 0:
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_threshold_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_threshold_idx]
        else:
            optimal_threshold = 0.5
        
        y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
        val_f1 = f1_score(y_val_knn, y_val_pred)
        
        print(f"k={k}, metric={metric}: F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = {'k': k, 'metric': metric, 'threshold': optimal_threshold}
            best_model = knn
            best_scaler = knn_scaler

print(f"\nBest KNN parameters: {best_params}")

# Final evaluation on test set
X_test_knn_scaled = best_scaler.transform(X_test_knn)
y_test_proba = best_model.predict_proba(X_test_knn_scaled)[:, 1]
y_test_pred = (y_test_proba >= best_params['threshold']).astype(int)

# Calculate final metrics
test_f1 = f1_score(y_test_knn, y_test_pred)
test_precision = precision_score(y_test_knn, y_test_pred)
test_recall = recall_score(y_test_knn, y_test_pred)
test_accuracy = accuracy_score(y_test_knn, y_test_pred)

print(f"\nKNN Test Performance:")
print(f"F1: {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")

# Save results
df_full['knn_straggler_probability'] = np.nan
df_full['knn_straggler_prediction'] = np.nan

test_indices = X_test_temp.index
df_full.loc[test_indices, 'knn_straggler_probability'] = y_test_proba
df_full.loc[test_indices, 'knn_straggler_prediction'] = y_test_pred

# Save final results
results_columns = [
    'inst_id', 'job_name', 'task_name', 'duration', 'predicted_duration',
    'is_straggler', 'knn_straggler_probability', 'knn_straggler_prediction',
    'absolute_error'
]

final_results = df_full[results_columns].copy()
final_results.to_csv('fixed_model_results.csv', index=False)

print(f"\nPipeline completed in {time.time() - total_start:.2f} seconds")
print("Results saved to 'fixed_model_results.csv'")
