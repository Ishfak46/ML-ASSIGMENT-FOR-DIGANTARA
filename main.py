
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Step 1: Data Preprocessing
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Datetime'])
    return data

#the file is named 'SMA_data.csv'
file_path = 'SMA_data.csv'
data = load_data(file_path)
sma = data['SMA'].values

# Generate synthetic labels for the existing dataset (since we don't have real maneuver labels)
def generate_synthetic_labels(sma, n_maneuvers=10, maneuver_duration=10):
    labels = np.zeros(len(sma))
    maneuver_indices = np.random.choice(np.arange(100, len(sma)-100), size=n_maneuvers, replace=False)
    for idx in maneuver_indices:
        sma[idx:idx+maneuver_duration] += np.random.normal(50, 10, maneuver_duration)
        labels[idx:idx+maneuver_duration] = 1
    return labels

labels = generate_synthetic_labels(sma)

# Step 2:Feature Extraction
def extract_features_and_labels(sma, labels, window_size=5):
    features = []
    new_labels = []
    for i in range(len(sma) - window_size + 1):
        window = sma[i:i+window_size]
        feature = [window.mean(), window.std(), window.max(), window.min()]
        features.append(feature)
        # Using the label at the center of the window
        new_labels.append(labels[i + window_size // 2])
    return np.array(features), np.array(new_labels)

features, adjusted_labels = extract_features_and_labels(sma, labels)

# Step 3: Maneuver Detection
X_train, X_test, y_train, y_test = train_test_split(features, adjusted_labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 4: Result Visualization
plt.figure(figsize=(10, 6))
plt.plot(data['Datetime'], sma, label='SMA')
maneuver_points = np.where(labels == 1)[0]
plt.scatter(data['Datetime'].iloc[maneuver_points], sma[maneuver_points], color='red', label='True Maneuvers')
plt.legend()
plt.title('SMA Variation with True Maneuvers')
plt.xlabel('Datetime')
plt.ylabel('Semi-Major Axis')
plt.show()

# Detect maneuvers in the entire dataset using the trained model
full_features, _ = extract_features_and_labels(sma, labels)
full_predictions = clf.predict(full_features)

plt.figure(figsize=(10, 6))
plt.plot(data['Datetime'], sma, label='SMA')
detected_maneuver_points = np.where(full_predictions == 1)[0]
plt.scatter(data['Datetime'].iloc[detected_maneuver_points], sma[detected_maneuver_points], color='green', label='Detected Maneuvers')
plt.legend()
plt.title('SMA Variation with Detected Maneuvers')
plt.xlabel('Datetime')
plt.ylabel('Semi-Major Axis')
plt.show()
