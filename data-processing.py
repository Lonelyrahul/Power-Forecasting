import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import psutil
import os

# Database connection
conn = pymysql.connect(host='localhost', user='root', password='Chennai@1234', database='power_house')
mycursor = conn.cursor()
mycursor.execute('SELECT * FROM dataset')
result = mycursor.fetchall()

# Define the column names in the same order as the MySQL table
columns = [
    'Date', 'Time', 'Global_active_power', 'Global_reactive_power',
    'Voltage', 'Global_intensity', 'sub_metering_1',
    'sub_metering_2', 'sub_metering_3', 'DateTime'
]

# Create DataFrame from DB result
df = pd.DataFrame(result, columns=columns)

# Preprocess the data
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=True)
df.drop(columns=['Date', 'Time'], inplace=True)
df = df.dropna(subset=['DateTime'])

# Convert all remaining columns to numeric
for col in df.columns:
    if col != 'DateTime':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# Add time-based features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['WeekDay'] = df['DateTime'].dt.weekday
df['Month'] = df['DateTime'].dt.month

# Rolling mean and peak hour features
df['Rolling_Mean'] = df['Global_active_power'].rolling(window=60).mean().fillna(method='bfill')
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
df['Peak_Hours'] = df['Hour'].apply(lambda x: 1 if 17 <= x <= 20 else 0)

# Prepare data for modeling
df_model = df.drop(columns='DateTime')
X = df_model.drop('Global_active_power', axis=1)
Y = df_model['Global_active_power']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=50, n_jobs=2, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
    'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# Function to monitor memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Memory in MB

# Train and evaluate models
result = {}
for name, model in models.items():
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    result[name] = {
        'RMSE': np.sqrt(mean_squared_error(Y_test, preds)),
        'MAE': mean_absolute_error(Y_test, preds),
        'R2_Score': r2_score(Y_test, preds),
        'Memory_MB': round(memory_usage(), 2)
    }

# Display results
res_df = pd.DataFrame(result).T
print("\nModel Evaluation Metrics:\n")
print(res_df)

# Optional: Save to CSV
res_df.to_csv('model_comparison_metrics.csv', index=True)