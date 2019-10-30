import sys
import numpy as np

def compute_error(true_vals, predicted_vals):
    '''
        Compute normalized RMSE
        Args:
            true_vals: numpy array of targets
            predicted_vals: numpy array of predicted values
    '''
    # Subtract minimum value
    min_value = np.min(true_vals)
    error = np.sum(np.square(true_vals-predicted_vals))/np.sum(np.square(true_vals-min_value))
    return error

target = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.int)[:, 0]
prediction = np.rint(np.genfromtxt(sys.argv[2]))
print(compute_error(target, prediction))
