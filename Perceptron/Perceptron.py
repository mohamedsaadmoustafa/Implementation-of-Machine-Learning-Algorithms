import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Perceptron:
    def __init__( self, learning_rate=0.1, iters=300, bias=True ):
        self.learning_rate = learning_rate
        self.iters = iters
        self.bias = bias
    
    def InitializeWeights( self, shape ):
        self.weights = np.random.uniform( 0, 1, shape + 1 ) # +1 for bias
        
    def UpdateWeights( self, xi, ti ):
        output = self.predict( xi )
        error = ti - output
        
        self.weights[1:] += self.learning_rate * error * xi
        if self.bias is False:         # no bais
            self.weights[0] = 0
        else:
            self.weights[0] += self.learning_rate * error   # bais
        return self.weights
    
    # training model
    def fit( self, x, t ):
        self.InitializeWeights( x.shape[1] )
        #print( 'w: ', self.weights )
        if self.bias is False:
            self.weights[0] = 0
        
        for _ in range( self.iters ):
            for xi, ti in zip( x, t ):
                self.UpdateWeights( xi, ti )

        return self
    
    def NetValue( self, x ):
        return np.dot( x, self.weights[1:] ) + self.weights[0]
    
    def predict( self, x ):
        net_val = self.NetValue( x )
        y_hat = np.sign( net_val )
        #print( net_val , '---',  y_hat )
        return y_hat
    
    def ReturnWeights( self ):
        return self.weights