import numpy as np
import Regression


class Ridge(Regression):
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
    
    def ridge(self, data, target):
        """
            Matrix Formulation of Ridge Model
            weights = (X^T . X)^-1 . X^T . y
        """
        ones = np.ones((len(data),1))
        data = np.hstack((ones, data))
        
        rows, cols = data.shape
        if rows >= cols == np.linalg.matrix_rank(data):
            # np.linalg.inv(X.T @ X + alpha * np.identity(X.shape[1]) @ (X.T @ y)
            I =  np.identity(cols)
            # set first 1 on the diagonal to zero so as not to include a bias term for the intercept
            I[0, 0] = 0
            # create a bias term corresponding to alpha for each column of X not including the intercept
            A = self.alpha * I
            w = np.linalg.inv(data.T.dot(data) + A ).dot(data.T).dot(target)
            self.coef_ = w[1:]
            self.intercept_ = w[0]
        else: self.normal(data, target) ###################################
    
    def fit( self, data_, target_ ):
        # apply some preprocessing methods
        data, target = self.preprocessing(data_, target_)
        # Choose model to fit
        self.ridge(data, target)
        return self
