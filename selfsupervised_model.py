import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('paysim dataset.csv')

# Drop irrelevant columns (assuming these columns exist based on the dataset you showed earlier)
df = df.drop(columns=['step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

# Check the distribution of 'isFraud'
print(df['isFraud'].value_counts())

# Scaling numerical features
scaler = StandardScaler()
df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = scaler.fit_transform(
    df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']])

# Split data into features (X) and target (y)
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Build the Autoencoder model
input_dim = X_train.shape[1]  # Number of features
encoding_dim = 14  # Latent space dimension

# Define autoencoder
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.models.Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=5, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Make predictions (reconstruction)
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error (MSE between input and output)
train_mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
test_mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Set a threshold based on the training MSE distribution
threshold = np.percentile(train_mse, 95)

# Flag anomalies: If MSE > threshold, flag as fraudulent
y_train_pred = train_mse > threshold
y_test_pred = test_mse > threshold

# Evaluate the model using classification metrics
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, test_mse)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Histogram of reconstruction errors
plt.figure(figsize=(10,6))
plt.hist(train_mse, bins=50, alpha=0.7, label='Training Error')
plt.hist(test_mse, bins=50, alpha=0.7, label='Test Error')
plt.axvline(x=threshold, color='r', linestyle='--', label="Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()
