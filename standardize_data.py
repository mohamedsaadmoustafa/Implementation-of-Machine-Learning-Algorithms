import numpy as np

def standardize_data(self, data):
    numerator = data - np.mean(data, axis=0)
    denominator = np.std(data, axis=0)
    return numerator / denominator
