import numpy as np
import Regression
from Optimization import SGD, Minibatch_Gradient_Decent

class RidgeCV(Regression):
    def __init__(
        self, 
        dtype="float64", learning_rate=0.01, 
        iters=300, normalize=False, 
        copy_X=True, method='normal', alpha=0.1, batch_size=32, tolerance=1e-07,
        is_shuffle=True, random_state=42, metric='mse', l2_penality=0.0001, optimizer='SGD'
    ):
        # invoking the __init__ of the Optimization class
        Regression.__init__(
            self, dtype, learning_rate, iters, normalize, 
            copy_X, method, alpha, batch_size, tolerance, is_shuffle, random_state, metric, optimizer
        )
        self.l2_penality = l2_penality
    
    def __train(self):
        if self.optimizer == "Minibatch_Gradient_Decent":
            opt = Minibatch_Gradient_Decent(
                data=self.data, target=self.target, 
                learning_rate=self.learning_rate, iters=self.iters, 
                tolerance=self.tolerance, 
                l2_penality=self.l2_penality, batch_size=self.batch_size
            )
        else:
            opt = SGD(
                data=self.data, target=self.target, 
                learning_rate=self.learning_rate, iters=self.iters, 
                tolerance=self.tolerance, 
                l2_penality=self.l2_penality
            )
        self.intercept_, self.coef_ = opt()

    def fit( self, data_, target_ ):
        # apply some preprocessing methods
        self.data, self.target = self.preprocessing(data_, target_)
        # model to fit
        self.__train()
        return self
    
    def best_l2(self, test_datat, test_target, alpha_range=10):
        """
            Search for the ideal regularization parameter on the validation data.
        """
        best_loss = 10e100
        alpha_list = [ l2*0.1 for l2 in range(1, alpha_range) ]
        for alpha in alpha_list:
            self.l2_penality = alpha
            self.fit(self.data, self.target)
            # Prediction on test set
            loss = self.score(test_datat, test_target)
            print(alpha, loss)

        if loss < best_loss:
            best_loss=loss
            self.best_l2_penality = alpha
        return self.best_l2_penality
    

    
