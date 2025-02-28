# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 06:00:59 2024

@author: khouloud Matri
"""

import pandas as pd
import numpy as np
from scipy.linalg import pinv
from sklearn.model_selection import KFold, train_test_split
from deap import  tools, algorithms
from fitnessFncError import fitness_fnc_error
from configure_GA import configure_ga
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math

import time

# Assuming y_true contains your true labels
# data = pd.read_excel('10_freq_trials2.xlsx').values
data = pd.read_excel('gestures_features2.xlsx').values

n_hid = 300  
#n_hid= int(input("Number of neurons at the hidden layer"))

predictors = data[:, 1:]
response = data[:, 0]

# predictors = data[:, :-1]
# response = data[:, -1]

# Split data into training+validation and test sets (e.g., 80% train+val, 20% test)
X_trainval, X_test, y_trainval, y_test = train_test_split(predictors, response, test_size=0.2, random_state=42)

# Set up K-fold cross-validation on the training+validation data
KFolds = 5

# kf = KFold(n_splits=KFolds, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=KFolds, shuffle=True, random_state=42)


# Initialize a list to store the confusion matrices or other evaluation metrics if needed
validation_scores = []

num_classes = 5
for fold, (train_index, test_index) in enumerate(skf.split(X_trainval, y_trainval)):

    
    # Training and validation split for the current fold
    train = X_trainval[train_index, :]
    train_label = y_trainval[train_index]

    validation = X_trainval[test_index, :]
    validation_label = y_trainval[test_index]
    
    # traindata = np.vstack((train, validation))
    # testdata = np.hstack((train_label, validation_label))

    # Create the input matrices for training and validation
    X = np.column_stack((train_label, train))              # Training features with labels
    S = train_label                                        # Training labels
    Xv = np.column_stack((validation_label, validation))   # Validation features with labels
    Sv = validation_label                                  # Validation labels


    # Xacc = X
    # Sacc = S

    # traindata = np.vstack((train, test))
    # testdata = np.hstack((train_label, test_label))

    ini = time.process_time()
    print(f"X shape {X.shape}")
    print(f'S shape {S.shape}')
    n_in = X.shape[1]
    n_out = S.shape[0]
    # n_out = 1
    # n_w = n_hid * (n_in + 1) + n_out * (n_hid + 1)


    N = X.shape[0]     # Number of training samples
    Nv = Xv.shape[0]   # Number of validation samples




    w2 = []
    # Initialize weights for the hidden and output layers(ELM)
    InputWeight = -0.1 + 0.2 * np.random.rand(n_hid, n_in + 1)
    OutputWeight = -0.1 + 0.2 * np.random.rand(n_out, n_hid + 1)
    
    # Hidden layer output matrix H (ELM)
    H = np.hstack([np.tanh(np.dot(np.hstack([X, np.ones((N, 1))]), InputWeight.T)), np.ones((N, 1))])



    #Genetic Algorithm (pruning):
    lambda_0 = np.zeros(H.shape[1])

    toolbox, numberOfVariables = configure_ga(lambda_0)

    FitnessFunction = lambda lambda_values: (fitness_fnc_error(H, lambda_values, S, Xv, Nv, InputWeight, Sv),)

    toolbox.register("evaluate", FitnessFunction)

    #Evolution and Optimization:
    population = toolbox.population(n=100)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, verbose=True) 
    #choosing best individual
    best_individual = tools.selBest(population, 1)[0]
    lambda_final = np.array(best_individual)
    
    num_neurons_best_model = np.sum(lambda_final > 0)
    print(f'num_neurons: {num_neurons_best_model} neurons')

    
    #Compute Output Weights and Evaluate Accuracy:
    L = np.eye(H.shape[1])
    for i in range(len(lambda_final)):
        L = lambda_final[i] * np.eye(H.shape[1])

    OutputWeight = np.dot(pinv(np.dot(H.T, H) + L), np.dot(H.T, S))

    S_estimado = np.dot(H, OutputWeight)
    # RMSE  = qmean(S - S_estimado)
   
    
    # Step 1: Round the continuous predicted values
    S_estimtrain = np.round(S_estimado).astype(int)

    # Step 2: Clip the values to ensure they fall within the valid class range (1 to 5)
    S_estimtrain = np.clip(S_estimtrain, 1, 5)

    # Ensure that S (true labels) are also integers
    S = S.astype(int)

    correct_train = np.sum(S_estimtrain == S)
    
    total = S.shape[0]
    TrainingAccuracy = correct_train/total
    print(f"Training accuracy {TrainingAccuracy}")
    fini = time.process_time()
    training_time = fini - ini
    print(f'Training time: {training_time} seconds')
    
    
    
    
    
    # Validation phase
    Hv = np.hstack([np.tanh(np.dot(np.hstack([Xv, np.ones((Nv, 1))]), InputWeight.T)), np.ones((Nv, 1))])
    S_estimval = np.dot(Hv, OutputWeight)

    
    # Step 1: Round the continuous predicted values
    S_estimvalid = np.round(S_estimval).astype(int)

    # Step 2: Clip the values to ensure they fall within the valid class range (1 to 5)
    S_estimvalid = np.clip(S_estimvalid, 1, 5)

    # Ensure that S (true labels) are also integers
    Sv = Sv.astype(int)
    
    # Calculate validation accuracy
    correct_val = np.sum(S_estimvalid == Sv)
    ValidationAccuracy = correct_val / Sv.shape[0]
    print(f"Validation accuracy: {ValidationAccuracy}")
    

#confusion matrix
class_labels = ['gesture 1', 'gesture 2', 'gesture 3', 'gesture 4', 'gesture 5']
# Generate the confusion matrix
cm = confusion_matrix(Sv, S_estimvalid)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot without seaborn
fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.matshow(cm_normalized, cmap='Blues')

# Add color bar
plt.colorbar(cax)

# Add annotations
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, f'{cm_normalized[i, j]:.2%}', ha='center', va='center', color='black', fontsize=12)

# Set ticks and labels
ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_normalized_no_seaborn.svg', bbox_inches='tight', format='svg', dpi=300)
plt.show()   

   



    
#New data  
initest = time.process_time()

 # Testing phase: Use the final model on the test set (held-out data)  
Xt = np.column_stack((y_test, X_test))              
St = y_test    
Nt = Xt.shape[0] 
 # X_test = np.column_stack((test_label, test))
Hv = np.hstack([np.tanh(np.dot(np.hstack([Xv, np.ones((Nv, 1))]), InputWeight.T)), np.ones((Nv, 1))])

H_test = np.hstack([np.tanh(np.dot(np.hstack([Xt, np.ones((Nt, 1))]), InputWeight.T)), np.ones((Nt, 1))])


S_estimtest = np.dot(H_test, OutputWeight)             #S_estimtest is the estimated output for the test set after applying the learned OutputWeight
S_estimtest = np.round(S_estimtest).astype(int)                     #predicted class labels

S_estimtest = np.clip(S_estimtest, 1, 5)
# Ensure that y_test (true test labels) are also integers
y_test = y_test.astype(int)

# Calculate test accuracy
correct_test = np.sum(S_estimtest == y_test)
TestAccuracy = correct_test / y_test.shape[0]
print(f"\nFinal Test accuracy: {TestAccuracy}")

finitest = time.process_time()
testing_time = finitest - initest


print(f'Testing time: {testing_time} seconds')


#confusion matrix
class_labels = ['gesture 1', 'gesture 2', 'gesture 3', 'gesture 4', 'gesture 5']
# Generate the confusion matrix
cm = confusion_matrix(y_test, S_estimtest)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot without seaborn
fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.matshow(cm_normalized, cmap='Greens')

# Add color bar
plt.colorbar(cax)

# Add annotations
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, f'{cm_normalized[i, j]:.2%}', ha='center', va='center', color='black', fontsize=12)

# Set ticks and labels
ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_normalized_no_seaborn.svg', bbox_inches='tight', format='svg', dpi=300)
plt.show()

