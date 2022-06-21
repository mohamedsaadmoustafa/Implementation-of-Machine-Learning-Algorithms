import numpy as np
import Regression
from Optimization import SGD, Minibatch_Gradient_Decent

class ElasticNet(Regression):
    def __init__(
        self, 
        dtype="float64", learning_rate=0.01,
        iters=300, normalize=False, 
        copy_X=True, method='normal', alpha=0.1, batch_size=32, tolerance=1e-07,
        is_shuffle=True, random_state=42, metric='mse', l1_penality=0.001, l2_penality=0.001, optimizer='SGD'
    ):
        # invoking the __init__ of the Optimization class
        Regression.__init__(
            self, dtype, learning_rate, iters, normalize, 
            copy_X, method, alpha, batch_size, tolerance, is_shuffle, random_state, metric, optimizer
        )
        self.l1_penality = l1_penality
        self.l2_penality = l2_penality
            
    def fit( self, data_, target_ ):
        # apply some preprocessing methods
        data, target = self.preprocessing(data_, target_)
        # model to fit
        if self.optimizer == "Minibatch_Gradient_Decent":
            opt = Minibatch_Gradient_Decent(
                data=data, target=target, 
                learning_rate=self.learning_rate, iters=self.iters, 
                tolerance=self.tolerance, 
                l1_penality=self.l1_penality, l2_penality=self.l2_penality, batch_size=self.batch_size
            )
        else:
            opt = SGD(
                data=data, target=target, 
                learning_rate=self.learning_rate, iters=self.iters, 
                tolerance=self.tolerance, 
                l1_penality=self.l1_penality, l2_penality=self.l2_penality
            )
            
        self.intercept_, self.coef_ = opt()
        return self
