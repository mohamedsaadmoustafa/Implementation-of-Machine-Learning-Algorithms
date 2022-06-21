import numpy as np

class Optimization:
    def __init__(self, data, target, learning_rate=0.0001, iters=1000, tolerance=1e-07, l1_penality=0, l2_penality=0, classify=False, lr_schedule='exponential'):
        self.data = data
        self.target = target
        self.learning_rate = learning_rate
        self.iters = iters
        self.tolerance = tolerance
        self.l1_penality = l1_penality
        self.l2_penality = l2_penality
        self.classify = classify
        self.lr_schedule = lr_schedule # 'time-based' 'step-based'  'exponential' 'polynomial' 
        self.cost_ = 0
        self.rows, self.cols = self.data.shape
        self.InitializeWeights(self.cols)
      
    def predict(self, data):
        y_hat = np.dot(data, self.coef_) + self.intercept_
        if self.classify: y_hat = Activation_Function(y_hat).sigmoid()
        return y_hat
        
    def gradient(self, data, target):
        try: rows = data.shape[0]
        except: rows = 1
        # Making predictions 
        predicted = self.predict(data) # Calculating the predicted values
        loss = target - predicted      # Calculating the individual loss for all the inputs
        l1 = ( self.l1_penality*np.sign(self.coef_) ) # l1 regularization
        l2 = ( 2 * self.l2_penality * self.coef_ )    # l2 regularization
        self.gradient_coef = (-2/rows) * ( np.dot(data.T, loss) - l1 + l2 )  # Calculating gradient
        self.gradient_intercept = (-2/rows) * np.sum(loss) # Calculating gradient
        
    def InitializeWeights(self, cols):
        self.intercept_ = np.random.uniform(0, 1, 1)
        self.coef_ = np.random.uniform(0, 1, cols)
    
    def cost( self, actual, predicted ):
        """
            using mean square error as cost function
        """
        cost = RegresstionScore(actual, predicted)(metric = 'mse')
        #if self.classify: cost = ClassificationScore(actual, predicted)(metric = 'Crossentropy') # Crossentropy F1_Score
        return cost
    
    def update_weights(self):
        # Updating the weights
        self.coef_ -= (self.learning_rate * self.gradient_coef) 
        self.intercept_ -=  (self.learning_rate * self.gradient_intercept) # Updating the bias

class Gradient_Decent(Optimization):
    def __init__(
        self, data, target, learning_rate=0.0001, 
        iters=1000, tolerance=1e-07, batch_size=32, 
        l1_penality=0, l2_penality=0, classify=False
    ):
        # invoking the __init__ of the Optimization class
        Optimization.__init__(self, data, target, learning_rate, iters, tolerance, l1_penality, l2_penality, classify)
        self.batch_size = batch_size
    
    def __call__(self):
        """
            Gradient Descent: 
        """
        #print(len(mini_batches), mini_batches[0].shape)
        for i in range( self.iters ):
            self.gradient(self.data, self.target)
            # Updating the weights
            self.update_weights()
            
            new_cost_ = self.cost(self.predict(self.data), self.target)
            ## stopping criteria: Check if the change in gradient is <= tolerance then break
            #if np.abs( self.cost_ - new_cost_) <= self.tolerance: 
            #    print(f'break at iteration #{i}')
            #    break;
            # save cost
            self.cost_ = new_cost_
        return self.intercept_, self.coef_
    
class Minibatch_Gradient_Decent(Optimization):
    def __init__(
        self, data, target, learning_rate=0.0001, 
        iters=1000, tolerance=1e-07, 
        batch_size=32, l1_penality=0, l2_penality=0, classify=False
    ):
        # invoking the __init__ of the Optimization class
        Optimization.__init__(self, data, target, learning_rate, iters, tolerance, l1_penality, l2_penality, classify)
        self.batch_size = batch_size

    def create_mini_batches(self, data, target):
        """
            A function to create a list containing mini-batches
        """
        mini_batches = []
        rows = len(data)
        n_minibatches = rows // self.batch_size
        i = 0
        for i in range(n_minibatches + 1):
            x = data[i * self.batch_size : (i + 1) * self.batch_size]
            y = target[i * self.batch_size : (i + 1) * self.batch_size]
            mini_batches.append((x, y))
        if rows % self.batch_size != 0:
            x = data[i * self.batch_size : rows]
            y = target[i * self.batch_size : rows]
            mini_batches.append((x, y))
        return mini_batches
    
    def __call__(self):
        """
            Mini-Batch Gradient Descent: 
            Parameters are updated after computing the gradient of error 
            with respect to a subset of the training set
        """
        mini_batches = self.create_mini_batches(self.data, self.target)
        #print(len(mini_batches), mini_batches[0].shape)
        for i in range( self.iters ):
            for data_batch, target_batch in mini_batches:
                self.gradient(data_batch, target_batch)
                # Updating the weights
                self.update_weights()
            new_cost_ = self.cost(self.predict(self.data), self.target)
            ## stopping criteria: Check if the change in gradient is <= tolerance then break
            #if np.abs( self.cost_ - new_cost_) <= self.tolerance: 
            #    print(f'break at iteration #{i}')
            #    break;
            # save cost
            self.cost_ = new_cost_
        return self.intercept_, self.coef_
    
class SGD(Optimization):
    def __init__(
        self, data, target, learning_rate=0.0001, 
        iters=1000, tolerance=1e-07, batch_size=32, 
        l1_penality=0, l2_penality=0, classify=False
    ):
        # invoking the __init__ of the Optimization class
        Optimization.__init__(self, data, target, learning_rate, iters, tolerance, l1_penality, l2_penality, classify)
        self.batch_size = batch_size
    
    def __call__(self):
        """
            S Gradient Descent: 
        """
        #print(len(mini_batches), mini_batches[0].shape)
        for i in range( self.iters ):
            indices = np.random.choice(self.data.shape[0], self.batch_size)
            data_, target_ = self.data[indices], self.target[indices]
            self.gradient(data_, target_)
            # Updating the weights
            self.update_weights()
            
            new_cost_ = self.cost(self.predict(self.data), self.target)
            ## stopping criteria: Check if the change in gradient is <= tolerance then break
            #if np.abs( self.cost_ - new_cost_) <= self.tolerance: 
            #    print(f'break at iteration #{i}')
            #    break;
            # save cost
            self.cost_ = new_cost_
        return self.intercept_, self.coef_
    
    
class Adagrad(Optimization): pass
# Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
class Adadelta(Optimization): pass
# Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
class RMSProp(Optimization): pass
# RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
class adam(Optimization): pass
# Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
