

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Lire le fichier CSV
dataf = pd.read_csv('/Users/khouloud Matri/Downloads/pfe/Nouveau dossier (9)/dynamic_grasping.csv',sep=',')


# Sampling frequency
Fs = 392  # Hz sampling rate

# Constants
w = 60  # Size of window for correlation coefficient

count = 0
#read data
for index, row in dataf.iterrows():
    row = row.dropna()
    
    r = []
    onset_index = []
    signal_buffer = []
    tkeo_buffer = []
    n = -1                                                            
    k = 0
    mE = 0
    
    t = np.linspace(0, len(tkeo_buffer), len(tkeo_buffer))/ Fs
    t = np.linspace(0, len(row), len(row))/ Fs
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(t, row)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EMG signal number:' + str(index+1) )
    plt.grid(True)
   
    count = 0
    #store signal
    for element in row:
        signal_buffer.append(element) # Store the signal in the buffer
        n += 1
        if n >= 2: #len(signal_buffer) >= 3:
            tkeo = signal_buffer[-2]**2 - signal_buffer[-3] * signal_buffer[-1]
            tkeo_buffer.append(tkeo)
            # Calculate correlation coefficient
            if n> 2*w+2: # len(tkeo_buffer) >= 2 * w:
                tkeo_past = np.abs(tkeo_buffer[-2 * w-2:-w-1])
                tkeo_future = np.abs(tkeo_buffer[-w-1:]) 
                r.append(np.corrcoef(tkeo_past, tkeo_future)[0, 1]) # corr_coeff
                # check for zero crossing
                if abs(r[-1]) <= 0.01:  # Change in state detected
                    onset_index.append(n-w)
                    if k == 0:
                        onset = onset_index[-1] # or simply n
                    k = k + 1#first onset
                    if k > 3: # >2 ? 
                        E1 = sum(np.abs(tkeo_buffer[onset_index[-3]:onset_index[-2]])) # initial.Energy
                        E2 = sum(np.abs(tkeo_buffer[onset_index[-2]:onset_index[-1]])) # current.Energy
                        previousE = sum(np.abs(tkeo_buffer[onset_index[-4]:onset_index[-3]]))
                        if mE < E1:
                            mE = E1
                            if E1 > 2*previousE and ():
                                onset = onset_index[-3] # k-2
                                ty = onset_index[-1] ########## -1
                                #plt.axvline(x=onset/Fs, color='blue',)
                            else:
                                onset = onset_index[-2] # k-1
                                ty = onset_index[-4] # k-3
                                #plt.axvline(x=onset/Fs, color='blue',)
                            s = 1 # Activity In Progress
                        # if onset == 0:
                        if (E2 < E1 and E2 > mE*0.05):
                            offset = onset_index[-1]; # k
                            s = 0  #  activityinProgress
                        else: 
                             offset = ty
                        if (s==1 and n - onset > Fs/2): 
                            s = 1 # Activity In Progress
                        elif (offset - onset > Fs/2):
                            # x(onset : offset) # is the detected Activity Region
                            plt.axvline(x=onset/Fs, color='green', linestyle='--', label='Vertical line at x=3')
                            plt.axvline(x=offset/Fs, color='red', linestyle='--', label='Vertical line at x=3')
                            mE=0
                            k = 0
                            onset = 0
                            onset_index = []
           
    #signalplot
    #t = np.linspace(0, len(tkeo_buffer), len(tkeo_buffer))/ Fs
    #axs[1].plot(t, tkeo_buffer,color='red')
    axs[1].plot(t[1:-1], tkeo_buffer,color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('TKE_Energy')
    plt.title('EMG signal number:' + str(index+1) )
    plt.grid(True)
    plt.show()
    
    

    
                                             

        



