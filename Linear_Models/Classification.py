import numpy as np
import linear_models
from Evaluation metrics import ClassificationScore
from Activation_Functions import Activation_Function

class Classification(linear_models):
    def __init__( 
        self, 
        dtype="float64", learning_rate=0.01,
        iters=300, normalize=False, 
        copy_X=True, method='normal', alpha=0.1, batch_size=32, tolerance=1e-07,
        is_shuffle=True, random_state=42, metric='mse', optimizer='SGD'
    ):    
        # invoking the __init__ of the Optimization class
        linear_models.__init__(
            self, dtype, learning_rate, iters, normalize, 
            copy_X, method, alpha, batch_size, tolerance, is_shuffle, 
            random_state, metric, optimize
        )
    
    def predict_prob( self, test_data_ ):
        if self.normalize: 
            test_data = self.normalize_2d(test_data_)
        if self.copy_X: 
            test_data = test_data_.copy()
        else: 
            test_data = test_data_
        # apply model to predict inserted data
        return test_data.dot(self.coef_) + self.intercept_
        
    def predict( self, test_data_ ): # y_hat
        # apply model to predict inserted data
        y_hat = self.predict_prob(test_data_)
        return Activation_Function(y_hat).sigmoid()
    
    def score(self, test_data, test_target):
        predict_target = self.predict(test_data)
        return ClassificationScore(test_target, predict_target)(metric=self.metric)
