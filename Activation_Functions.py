import numpy as np 

class Activation_Function:
    def __init__(self, x):
        self.x = np.array(x)
        
    def sigmoid(self):
        return 1.0 / (1 + np.exp(-self.x))

    def sigmoid_derivative(self):
        return self.x * (1.0 - self.x)
    
    def __call__(self, method):
        if method == 'sigmoid': return self.sigmoid(),
        elif method == 'sigmoid_derivative': return self.sigmoid_derivative()
        else: raise ValueError
            
#Activation_Function(0.1)('sigmoid')
Activation_Function(0.1).sigmoid()
