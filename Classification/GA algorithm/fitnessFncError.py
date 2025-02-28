import numpy as np
from qmean import qmean

def fitness_fnc_error(H, lambda_values, S, Xv, Nv, InputWeight, Sv):
    L = np.eye(H.shape[1])
    for l in lambda_values:
        L = l * np.eye(H.shape[1])
    OutputWeight = np.linalg.pinv(H.T @ H + L) @ H.T @ S
    Hv = np.tanh(np.hstack((Xv, np.ones((Nv, 1)))) @ InputWeight.T)
    Hv = np.hstack((Hv, np.ones((Nv, 1))))
    S_estimado = Hv @ OutputWeight
    ValidatingAccuracy = qmean(Sv - S_estimado)
    fitness = ValidatingAccuracy
    return fitness
