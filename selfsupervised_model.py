import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset (Assume the dataset like the Kaggle Fraud Dataset)
df = pd.read_csv('paysim dataset.csv')  # Load a CSV file, replace with dataset path
df = df.drop(columns=['step','type','nameOrig','nameDest','isFlaggedFraud']) # Drop irrelevant columns
print(df)
print(df[["isFraud"]].value_counts()) # Distribution of fraudulent cases in the dataset

# Preprocess data (scaling all numerical features)
scaler = StandardScaler()
df['amount'] = scaler.fit_transform(df['amount'].values.reshape(-1, 1))
df['oldbalanceOrg'] = scaler.fit_transform(df['oldbalanceOrg'].values.reshape(-1, 1))
df['oldbalanceDest'] = scaler.fit_transform(df['oldbalanceDest'].values.reshape(-1, 1))
df['newbalanceDest'] = scaler.fit_transform(df['newbalanceDest'].values.reshape(-1, 1))
X = df.drop(columns=['isFraud'])  # Drop the target column ('isFraud') and irrelevant columns ('Time')
y = df['isFraud']

# Split the data into training and testing datasets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Convert the data to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Define a simple Autoencoder
input_dim = X_train.shape[1]  # Number of features (columns)
encoding_dim = 14  # Latent space dimension (you can experiment with this)

# Build the Autoencoder model
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.models.Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the Autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Make predictions (Reconstruction)
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error (mean squared error between input and output)
train_mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
test_mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Set a threshold to flag anomalies (fraudulent transactions)
threshold = np.percentile(train_mse, 95)  # You can adjust the percentile to control sensitivity

# Flag anomalies: If MSE is greater than threshold, flag as fraudulent
y_train_pred = train_mse > threshold
y_test_pred = test_mse > threshold

# Evaluate the performance using classification metrics
print("Train MSE threshold:", threshold)
print("\nClassification Report (Test Set):")
print(classification_report(y, y_test_pred))  # Compare predicted anomalies to actual fraud labels

# Confusion Matrix
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y, y_test_pred))

# Visualization of Reconstruction Error Distribution
plt.figure(figsize=(10,6))
plt.hist(train_mse, bins=50, alpha=0.7, label='Training Error')
plt.hist(test_mse, bins=50, alpha=0.7, label='Test Error')
plt.axvline(x=threshold, color='r', linestyle='--', label="Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()
