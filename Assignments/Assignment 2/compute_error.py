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
    #min_value = np.min(true_vals)
    count = 0
    for i in range(len(true_vals)):
        if true_vals[i] == predicted_vals[i]:
            count = count+1
    return count

target = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.int)[:, 0]
prediction = np.rint(np.genfromtxt(sys.argv[2]))
k = compute_error(target, prediction)
print(k)
p = (float)(k/(len(target)))
print(p)
