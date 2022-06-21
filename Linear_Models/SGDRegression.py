import numpy as np
import Regression
from Optimization import SGD


class SGDRegression(Regression):
    def __init__(
        self, 
        dtype="float64", learning_rate=0.01,
        iters=300, normalize=False, 
        copy_X=True, method='normal', alpha=0.1, batch_size=32, tolerance=1e-07,
        is_shuffle=True, random_state=42, metric='mse'
    ):
        # invoking the __init__ of the Optimization class
        Regression.__init__(
            self, dtype, learning_rate, iters, normalize, 
            copy_X, method, alpha, batch_size, tolerance, is_shuffle, random_state, metric
        )
        
    def __normal(self, data, target):
        """
            Ordinary Least Squares
        """
        # Minibatch_Gradient_Decent, SGD
        opt = SGD(
            data=data, target=target, 
            learning_rate=self.learning_rate, 
            iters=self.iters, tolerance=self.tolerance
        )#, self.batch_size)
        self.intercept_, self.coef_ = opt()
            
    def fit( self, data_, target_ ):
        # apply some preprocessing methods
        data, target = self.preprocessing(data_, target_)
        # model to fit
        self.__normal(data, target) 
        return self
