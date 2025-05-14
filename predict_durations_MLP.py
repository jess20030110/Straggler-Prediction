fimport pandas as pd
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

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
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
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
df = pd.read_csv("filtered_grouped_data.csv")
print(f"Loaded {len(df):,} rows")
print("Columns:", df.columns.tolist())

# Convert submit_time to datetime if not already
df['submit_time'] = pd.to_datetime(df['submit_time'], unit='s') 
# Verify the conversion
print(df['submit_time'].head())

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


# Handle missing GPU data
drop_cols = [
    'machine_cpu', 'machine_cpu_iowait', 'workload', 
    'machine_num_worker', 'machine_cpu_kernel', 
    'machine_load_1', 'machine_gpu', 'machine_cpu_usr',
    'machine_net_receive', 'gpu_type', 'gpu_type_spec'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

moderate_nan_cols = [
    'cpu_usage', 'avg_mem', 'avg_gpu_wrk_mem', 
    'max_gpu_wrk_mem', 'gpu_wrk_util', 'max_mem',
    'read_count', 'write_count', 'read', 'write',
    'plan_gpu'
]
# Fill with 0 if NaN indicates "no usage" (e.g., GPU metrics for CPU-only tasks)
df[moderate_nan_cols] = df[moderate_nan_cols].fillna(0)

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

# Prepare final datasets
X_train = X_train_temp[features].fillna(0).astype('float32')
X_val = X_val_temp[features].fillna(0).astype('float32')
X_test = X_test_temp[features].fillna(0).astype('float32')

print("\n=== Feature Summary ===")
print(f"Total features: {len(features)}")
print("Numeric features stats:")
print(X_train.describe())

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

# # Define loss and optimizer
# criterion = nn.MSELoss()  # Using MSELoss for regression
criterion = nn.SmoothL1Loss() #HuberLoss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Early stopping parameters
best_loss = float('inf')
patience = 10
counter = 0
early_stop = False

# Training loop
num_epochs = 100
print("\nTraining progress:")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    
    # Validation - now uses validation set instead of test set
    model.eval()
    with torch.no_grad():
        X_val_tensor_device = X_val_tensor.to(device)
        y_val_tensor_device = y_val_tensor.to(device)
        y_pred = model(X_val_tensor_device)
        val_loss = criterion(y_pred, y_val_tensor_device).item()
    
    # Update learning rate
    scheduler.step(val_loss)

    # Print statistics every 5 epochs
    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
    
    # Sample predictions every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            # Sample predictions using validation data
            sample_idx = torch.randint(0, len(X_val_tensor_device), (5,))
            sample_pred = model(X_val_tensor_device[sample_idx])
            sample_true = y_val_tensor_device[sample_idx]
            
            print("\nSample Predictions:")
            for i in range(5):
                pred = target_scaler.inverse_transform(sample_pred[i].cpu().numpy().reshape(1,-1))[0][0]
                true = target_scaler.inverse_transform(sample_true[i].cpu().numpy().reshape(1,-1))[0][0]
                print(f"Pred: {pred:.1f}s | True: {true:.1f}s | Error: {abs(pred-true):.1f}s")
                
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_mlp_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            early_stop = True
            break

# Load best model
model.load_state_dict(torch.load('best_mlp_model.pth'))
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

# Straggler analysis
grouping_cols = ['job_name', 'task_name']
df_all['avg_duration'] = df_all.groupby(grouping_cols)['duration'].transform('mean')
df_all['straggler_threshold'] = df_all['avg_duration'] * 1.5
df_all['is_straggler'] = df_all['duration'] > df_all['straggler_threshold']
df_all['pred_avg_duration'] = df_all.groupby(grouping_cols)['predicted_duration'].transform('mean')
df_all['pred_straggler_threshold'] = df_all['pred_avg_duration'] * 1.5
df_all['pred_is_straggler'] = df_all['predicted_duration'] > df_all['pred_straggler_threshold']

output_columns = [
    'inst_id', 'user', 'plan_gpu', 'plan_cpu', 'plan_mem', 'task_name',
    'inst_number', 'duration', 'predicted_duration', 'absolute_error', 'absolute_percentage_error',
    'job_name', 'submit_time', 'group', 'wait_time', 'is_straggler', 'pred_is_straggler'
]

output_df = df_all[output_columns].rename(columns={'inst_id': 'job_id'})
output_file = "mlp_predictions_with_errors.csv"
output_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
print("="*50)
print("Final output columns:", output_df.columns.tolist())

# Visualization
import matplotlib.pyplot as plt
# Correct way to compare actual vs predicted values
plt.figure(figsize=(10,6))
plt.scatter(y_test_original, predictions_original, alpha=0.3)
plt.plot([min(y_test_original), max(y_test_original)], 
         [min(y_test_original), max(y_test_original)], 'r--')
plt.xlabel('True Duration (s)')
plt.ylabel('Predicted Duration (s)')
plt.title('True vs Predicted Durations')
plt.savefig('True_vs_Prediction.png')
plt.close()

# Error distribution
plt.figure(figsize=(10,6))
errors = y_test_original - predictions_original
plt.hist(errors, bins=50, alpha=0.75)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Prediction Error (s)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.savefig('Error_Distribution.png')
plt.close()

# Optional: Feature importance analysis using permutation importance
print_step("9. Feature importance analysis (optional)...")
try:
    from sklearn.inspection import permutation_importance
    
    # Create a simple wrapper for PyTorch model to use with sklearn
    class PyTorchModelWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            
        def predict(self, X):
            X_tensor = torch.FloatTensor(X).to(self.device)
            self.model.eval()
            with torch.no_grad():
                return self.model(X_tensor).cpu().numpy()
            
        # Add a dummy fit method to satisfy scikit-learn's requirements
        def fit(self, X, y):
            pass  # No training is performed here

        # Add a score method to calculate R²
        def score(self, X, y):
            predictions = self.predict(X)
            return r2_score(y, predictions)
    
    wrapped_model = PyTorchModelWrapper(model, device)
    
    # Calculate permutation importance
    result = permutation_importance(wrapped_model, X_test_scaled, y_test_transformed, 
                                   n_repeats=5, random_state=42)
    
    # Get importance scores
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    # Save feature importance to file
    importance_df.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to 'feature_importance.csv'")
    
    # Plot feature importance
    plt.figure(figsize=(12,8))
    plt.barh(importance_df['Feature'].head(15)[::-1], importance_df['Importance'].head(15)[::-1])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('Feature_Importance.png')
    plt.close()
    
except Exception as e:
    print(f"Could not calculate feature importance: {e}")
