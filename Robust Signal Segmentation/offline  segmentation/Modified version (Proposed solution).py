# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:45:58 2024

@author: khouloud Matri
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Lire le fichier CSV
dataf = pd.read_csv('/Users/khouloud Matri/Downloads/pfe/Nouveau dossier (9)/dynamic_grasping.csv',sep=',')


# Sampling frequency
Fs = 392  # Assuming 1000 Hz sampling rate

# Constants
w = 80 # Size of window for correlation coefficient

#read data
for index, row in dataf.iterrows():
    row = row.dropna()
    onset = 0
    r = []  
    onset_index     = [0] #to check
    onset_buffer    = []  #sure
    offset_index    = [0] # to check
    offset_buffer   = [] #sure 
    signal_buffer   = []
    tkeo_buffer     = []
   
    n = -1   #number of read samples                                                     
    s = 0    # no activity (1 means activity)
    
    t = np.linspace(0, len(row), len(row))/ Fs
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    axs[0].plot(t, row)
    axs[0].grid(True)
    axs[4].plot(t, row)
    axs[4].grid(True)
    axs[3].plot(t, row)
    axs[3].grid(True)
    
    #store signal
    for element in row:
        signal_buffer.append(element) # Store the signal in the buffer
        n += 1
        #offset = 0
        if n >= 2: #len(signal_buffer) >= 3:
            tkeo = signal_buffer[-2]**2 - signal_buffer[-3] * signal_buffer[-1]
            tkeo_buffer.append(tkeo)
            if n > 2*w+2:                # len(tkeo_buffer) >= 2 * w:
                tkeo_past = np.abs(tkeo_buffer[-2 * w-2:-w-1])
                tkeo_future = np.abs(tkeo_buffer[-w-1:]) 
                r.append(np.corrcoef(tkeo_past, tkeo_future)[0, 1]) # corr_coeff
                                        # check for zero crossing
                if abs(r[-1]) <= 0.01:  # Change in state detected
                    axs[3].axvline(x = (n-w) /Fs, color='green', linestyle='--', label='Vertical line at x=3')
                    E1 = sum(np.abs(tkeo_buffer[n-w-w:n-w])) # initial.Energy
                    E2 = sum(np.abs(tkeo_buffer[n-w:n])) # current.Energy
                    if E2 > 2 * E1 : # and E1 > E2*0.05: # and s == 0:
                        onset_index.append(n-w)
                        if s==0:
                            onset_buffer.append(n-w)
                            axs[4].axvline(x = onset_buffer[-1] /Fs, color='green', linestyle='--')
                            s = 1
                    if E1 > 2 * E2: # and E1 > E2*0.05: # and s == 0:
                        offset_index.append(n-w)
                        if s == 1:
                            if (n-w-onset_buffer[-1]) > Fs/2:
                                offset_buffer.append(n-w)
                                s = 0
                                axs[4].axvline(x = offset_buffer[-1] /Fs, color='red', linestyle='--')
                            

     
    #axs[1].plot(tkeo_buffer,color='red')
    axs[1].plot(t[1:-1],tkeo_buffer,color='red')
    axs[1].grid(True)
    axs[2].plot(t[w+2:-w-1],r,color='green')
    axs[2].grid(True)
    plt.xlabel('Time (s)')
    plt.show()
    
