import pandas as pd
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, TargetEncoder, QuantileTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import KFold
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def print_step(message):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")

# --- Configuration ---
BASE_PATH = "/mnt/newdisk/jess/Alibaba2020"
PREPROCESS_PATH = f"{BASE_PATH}/Preprocess"
RESULT_PATH = f"{BASE_PATH}/Result"
# The output of this script is a single CSV file
OUTPUT_CSV_PATH = f"{RESULT_PATH}/predictions_for_classification.csv"

# Track total execution time
total_start = time.time()

# 0. GPU Setup
print_step("0. Initializing GPU...")
GPU_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda" if GPU_AVAILABLE else "cpu")
print(f"{'⚡ GPU: ' + torch.cuda.get_device_name(0) + ' detected!' if GPU_AVAILABLE else '⚠️ GPU not found - Falling back to CPU'}")

# Define the MLP model (unchanged)
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_scale=6):
        super(MLPRegressor, self).__init__()

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
df = pd.read_csv(f"{PREPROCESS_PATH}/instance_preprocessed.csv")
print(f"Loaded {len(df):,} rows")

# Time-based feature engineering
print_step("2. Performing feature engineering and One-hot encoding...")
df['submit_time'] = pd.to_datetime(df['submit_time'], unit='s')
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

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['gpu_type_machine_spec'], prefix='gpu_type')
df = pd.get_dummies(df, columns=['task_name'], prefix='task_name')
df = pd.get_dummies(df, columns=['gpu_name'], prefix='gpu_name')
df = df.reset_index(drop=True)

# 2. Define features and target
temporal_features = ['submit_hour_sin', 'submit_hour_cos',
                    'submit_dayofweek_sin', 'submit_dayofweek_cos',
                    'submit_dayofyear_sin', 'submit_dayofyear_cos',
                    'days_since_start']

base_features = ['plan_gpu', 'plan_mem', 'plan_cpu', 'cap_cpu', 'cap_mem', 'cap_gpu', 
                'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem','avg_gpu_wrk_mem', 'max_gpu_wrk_mem',
                'inst_number', 'read', 'write', 'read_count', 'write_count'] + \
               [col for col in df.columns if col.startswith('gpu_type')] + \
               [col for col in df.columns if col.startswith('task_name')] + \
               [col for col in df.columns if col.startswith('gpu_name')] + \
               temporal_features

base_features = [f for f in base_features if f in df.columns]
target = "duration"

# 3. Train/test split (80:20)
print_step("3. Creating train/test splits...")
df['original_group'] = df['group'].copy()
X_temp = df[base_features + ['user', 'group', 'machine']].copy()
y = df[target].astype('float32')

# Single split (80% train, 20% test)
X_train_temp, X_test_temp, y_train, y_test = train_test_split(
    X_temp, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
)
# Add a 'split' column to the original dataframe
df['split'] = 'train'
df.loc[X_test_temp.index, 'split'] = 'test'
# print(pd.qcut(y, q=5).value_counts())

print(f"Train size: {len(X_train_temp)} ({len(X_train_temp)/len(df)*100:.1f}%)")
print(f"Test size: {len(X_test_temp)} ({len(X_test_temp)/len(df)*100:.1f}%)")

# 4. Target encoding (now only applied to train/test to prevent leakage)
print_step("4. Target encoding and generate new statistical features...")
user_encoder = TargetEncoder(target_type='continuous', smooth='auto', cv=5, random_state=42)
group_encoder = TargetEncoder(target_type='continuous', smooth='auto', cv=5, random_state=42)
machine_encoder = TargetEncoder(target_type='continuous', smooth='auto', cv=5, random_state=42)

X_train_temp['user_encoded'] = user_encoder.fit_transform(X_train_temp[['user']], y_train).flatten()
X_train_temp['group_encoded'] = group_encoder.fit_transform(X_train_temp[['group']], y_train).flatten()
X_train_temp['machine_encoded'] = machine_encoder.fit_transform(X_train_temp[['machine']], y_train).flatten()

X_test_temp['user_encoded'] = user_encoder.transform(X_test_temp[['user']]).flatten()
X_test_temp['group_encoded'] = group_encoder.transform(X_test_temp[['group']]).flatten()
X_test_temp['machine_encoded'] = machine_encoder.transform(X_test_temp[['machine']]).flatten()

# New statistical features generation
for dataset in [X_train_temp, X_test_temp]:
    dataset['cpu_usage_ratio'] = dataset['cpu_usage'] / dataset['plan_cpu'].replace(0, 1)
    dataset['mem_usage_ratio'] = dataset['avg_mem'] / dataset['plan_mem'].replace(0, 1)
    dataset['gpu_usage_ratio'] = dataset['gpu_wrk_util'] / dataset['plan_gpu'].replace(0, 1)

features = base_features + ['user_encoded', 'group_encoded', 'machine_encoded',
                          'cpu_usage_ratio', 'mem_usage_ratio', 'gpu_usage_ratio']

X_train = X_train_temp[features].fillna(0).astype('float32')
X_test = X_test_temp[features].fillna(0).astype('float32')

# 5. Feature scaling (unchanged, now only train/test)
print_step("5. Feature scaling and Target scaling...")
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Target scaling
target_scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
y_train_transformed = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_transformed = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_transformed)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_transformed)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 6. MLP Training (unchanged, but remove validation checks)
print_step("6. Training MLP model...")
input_dim = X_train_scaled.shape[1]
model = MLPRegressor(input_dim, hidden_scale=6).to(device)

criterion = nn.L1Loss()  # MAE Loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Initialize KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
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
    
    # Cross-validation for early stopping (on training set)
    model.eval()
    cv_losses = []
    with torch.no_grad():
        for train_idx, val_idx in kfold.split(X_train_scaled):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train_transformed[train_idx], y_train_transformed[val_idx]
            
            X_fold_val_tensor = torch.FloatTensor(X_fold_val).to(device)
            y_fold_val_tensor = torch.FloatTensor(y_fold_val).to(device)
            
            y_pred = model(X_fold_val_tensor)
            fold_loss = criterion(y_pred, y_fold_val_tensor).item()
            cv_losses.append(fold_loss)
    
    avg_cv_loss = np.mean(cv_losses)
    scheduler.step(avg_cv_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, CV Loss: {avg_cv_loss:.4f}')
    
    # Early stopping based on CV loss
    if avg_cv_loss < best_val_loss:
        best_val_loss = avg_cv_loss
        patience_counter = 0
        os.makedirs(RESULT_PATH, exist_ok=True)
        torch.save(model.state_dict(), f'{RESULT_PATH}/mlp_model_hiddenScale6.pth')
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch+1} (CV loss did not improve for 10 epochs)')
            break

print("MLP Training Complete.")

# 7. MLP Evaluation 
print_step("7. Evaluating MLP model...")
model.load_state_dict(torch.load('/mnt/newdisk/jess/Alibaba2020/Result/mlp_model_hiddenScale6.pth'))
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    predictions = model(X_test_device).cpu().numpy()
    y_test_original = target_scaler.inverse_transform(y_test_transformed.reshape(-1, 1)).flatten()
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
r2 = r2_score(y_test_original, predictions_original)

print(f"MLP Test Performance:")
print(f"MAE: {mae:.2f} seconds")
print(f"RMSE: {rmse:.2f} seconds")
print(f"R² Score: {r2:.4f}")

valid_mask = y_test_original != 0
if np.any(valid_mask):
    ape = np.abs((y_test_original[valid_mask] - predictions_original[valid_mask]) / y_test_original[valid_mask]) * 100
    mape = np.mean(ape)
    print(f"MAPE (non-zero): {mape:.2f}%")
    print(f"Predictions within 25% of actual: {np.mean(ape <= 25)*100:.2f}%")

# 8. Generate Predictions for the ENTIRE Dataset
print_step("8. Generating predictions for the entire dataset...")
model.eval()
# Apply all transformations to the full dataframe `df`
df['user_encoded'] = user_encoder.transform(df[['user']]).flatten()
df['group_encoded'] = group_encoder.transform(df[['group']]).flatten()
df['machine_encoded'] = machine_encoder.transform(df[['machine']]).flatten()
df['cpu_usage_ratio'] = df['cpu_usage'] / df['plan_cpu'].replace(0, 1)
df['mem_usage_ratio'] = df['avg_mem'] / df['plan_mem'].replace(0, 1)
df['gpu_usage_ratio'] = df['gpu_wrk_util'] / df['plan_gpu'].replace(0, 1)

X_full = df[features].fillna(0).astype('float32')
X_full_scaled = scaler.transform(X_full)
X_full_tensor = torch.FloatTensor(X_full_scaled).to(device)

with torch.no_grad():
    full_predictions_transformed = model(X_full_tensor).cpu().numpy()
    # Inverse transform predictions to get original scale
    df['predicted_duration'] = target_scaler.inverse_transform(full_predictions_transformed.reshape(-1, 1)).flatten()
    df['absolute_error'] = np.abs(df['duration'] - df['predicted_duration'])
    df['absolute_percentage_error'] = np.where(
        df['duration'] != 0,
        np.abs((df['duration'] - df['predicted_duration']) / df['duration']) * 100,
        np.nan
    )

# 9. Save Augmented Data to a Single CSV File
print_step(f"9. Saving augmented data to {OUTPUT_CSV_PATH}...")
df.to_csv(OUTPUT_CSV_PATH, index=False)
print("Data saved successfully.")
print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
