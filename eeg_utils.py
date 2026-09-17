"""
Dataset A loading utilities.
"""

import os

import numpy as np
import scipy.io
from scipy import signal as sp_signal


def load_mat_data(mat_files: list, data_dir: str):
    """
    Loads and concatenates one or more S0X_EEG_MI.mat files.
    Returns X: (trials, channels, samples), Y: (trials, 1).
    """
    all_X, all_Y = [], []
    for mat_file in mat_files:
        filepath = os.path.join(data_dir, mat_file)
        mat_data = scipy.io.loadmat(filepath)
        X, Y = mat_data['X'], mat_data['Y']
        # Confirmed real raw layout requires this transpose 
        X = X.transpose(2, 1, 0)
        all_X.append(X)
        all_Y.append(Y)
    X_full = np.concatenate(all_X, axis=0)
    Y_full = np.concatenate(all_Y, axis=0)
    return X_full, Y_full



