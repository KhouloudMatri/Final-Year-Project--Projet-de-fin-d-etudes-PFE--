import numpy as np

def qmean(w):

    v = w.flatten()  
    n_v = len(v)     
    rms = np.sqrt(np.sum(v * v) / n_v)  
    return rms
