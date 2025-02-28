import numpy as np
import pandas as pd
import math
import os
from csv import reader
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.stats import skew, kurtosis

# Feature extraction functions
def rms(x):
    y = np.power(x, 2)
    return np.sqrt(np.mean(y))

def mav(x):
    y = np.absolute(x)
    return np.mean(y)

def wl(x):
    result = 0
    for i in range(1, len(x)):
        result += np.abs(x[i] - x[i-1])
    return result

def ssc(x):
    result = 0
    for i in range(1, len(x)-1):
        result += np.abs((x[i] - x[i-1]) * (x[i] - x[i+1]))
    return result

def var(x):
    n = len(x)
    mean = sum(x) / n
    deviations = [(i - mean) ** 2 for i in x]
    variance = sum(deviations) / n
    return variance

def stdev(x):
    variance = var(x)
    std_dev = math.sqrt(variance)
    return std_dev

def skewness(x):
    return skew(x)

def kurt(x):
    return kurtosis(x)

def I_EMG(x):
    return sum(abs(x))

def simple_square_integral(x):
    return sum(x**2)

def entropy(x):
    return scipy.stats.entropy(x)

def maxav(x):
    return max(abs(i) for i in x)

def wamp(x):
    wamp = 0
    for i in range(1, len(x)-1):
        wamp += np.abs(x[i] - x[i+1])
    return wamp

def minav(x):
    return min(abs(i) for i in x)

def zerocrossing(x):
    return (np.diff(np.sign(x)) != 0).sum()

# Initialize the scaler
scaler = MinMaxScaler()

# Path to the gestures folder
gestures_folder = r'C:\Users\khouloud Matri\Downloads\gestures'

all_features = []
labels = []

# Loop through each gesture folder
for gesture_label, gesture_folder in enumerate(os.listdir(gestures_folder)):
    gesture_folder_path = os.path.join(gestures_folder, gesture_folder)
    if os.path.isdir(gesture_folder_path):
        # Loop through each CSV file in the gesture folder
        for csv_file in os.listdir(gesture_folder_path):
            csv_file_path = os.path.join(gesture_folder_path, csv_file)
            if csv_file.endswith('.csv'):
                with open(csv_file_path, 'r') as read_obj1:
                    csv_reader1 = reader(read_obj1)
                    for row in csv_reader1:
                        # Convert row to float array
                        row1 = np.array([float(i) for i in row])
                        
                        features = [
                            rms(row1), mav(row1), wl(row1), ssc(row1), var(row1), stdev(row1),
                            skewness(row1), kurt(row1), I_EMG(row1), simple_square_integral(row1),
                            entropy(row1), maxav(row1), wamp(row1), minav(row1), zerocrossing(row1)
                        ]
                        
                        all_features.append(features)
                        labels.append(gesture_label)

# Convert to DataFrame
features_df = pd.DataFrame(all_features, columns=["RMS", "MAV", "WL", "SSC", "VAR", "SD", "SKEW", "KURT", "IEMG", "SSI", "EN", "MAXAV", "WAMP", "MINAV", "ZC"])
labels_df = pd.DataFrame(labels, columns=["Label"])

# Normalize features
normalized_features = scaler.fit_transform(features_df)
normalized_features_df = pd.DataFrame(normalized_features, columns=features_df.columns)

# Combine features and labels
final_df = pd.concat([normalized_features_df, labels_df], axis=1)

# Save to CSV
final_df.to_csv("gestures_features.csv", index=False, header=True)
