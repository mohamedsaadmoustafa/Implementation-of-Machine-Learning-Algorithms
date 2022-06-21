import numpy as np
import Regression

class LinearRegression(Regression):
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
        
    def solver(self, data, target):
        """
            Matrix Formulation of the Multiple Regression Model
            weights = (X^T . X)^-1 . X^T . y
            ε = Y − ̂ Y = [I − X (X.T . X)^−1 . X.T] Y
        """
        ones = np.ones((len(data),1))
        data = np.hstack((ones, data))
        rows, cols = data.shape
        if rows >= cols == np.linalg.matrix_rank(data):
            w = np.linalg.inv(data.T.dot(data)).dot(data.T).dot(target)
            self.coef_ = w[1:]
            self.intercept_ = w[0]
        else: print("Method=\'solver\' cannot be used: Data hasn't full column rank.")
            
    def qrsolver(self, data, target):
        """
            Solve via QR Decomposition
            breaking X (n x m) matrix down into its constituent elements.
            X is the matrix that we wish to decompose
            Q . R = qr(X)
            Q a matrix with the size m x m, and R is an upper triangle matrix with the size m x n.
            weights = R^-1 . Q.T . y
        """
        ones = np.ones((len(data),1))
        data = np.hstack((ones, data))
        
        Q, R = np.linalg.qr(data)
        w = np.linalg.inv(R).dot(Q.T).dot(target)
        self.coef_ = w[1:]
        self.intercept_ = w[0]
                
    def svd(self, data, target):
        """
            Solve via Singular-Value Decomposition
            U is a m x m matrix
            Sigma is an m x n diagonal matrix
            V^* is the conjugate transpose of an n x n matrix where * is a superscript.
            X^+ = U . D^+ . V^T
            weights = X^+ . y
        """
        ones = np.ones((len(data),1))
        data = np.hstack((ones, data))
        Q, R = np.linalg.qr(data)
        w = np.linalg.pinv(data).dot(target)
        self.coef_ = w[1:]
        self.intercept_ = w[0]
            
    def fit( self, data_, target_ ):
        # apply some preprocessing methods
        data, target = self.preprocessing(data_, target_)
        # Choose model to fit
        if self.method == 'solver': self.solver(data, target)
        elif self.method == 'qr': self.qrsolver(data, target)
        elif self.method == 'svd': self.svd(data, target)
        else: self.solver(data, target) 
        return self
