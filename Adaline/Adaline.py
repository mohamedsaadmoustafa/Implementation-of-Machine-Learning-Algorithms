import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Adaline(object):
    def __init__( self, learning_rate=0.1, iters=300, bias=True, mse_threshold=1 ):
        self.learning_rate = learning_rate
        self.iters = iters
        self.bias = bias
        self.mse_threshold = mse_threshold # stopping criteria
    
    def InitializeWeights( self, shape ):
        self.weights = np.random.uniform( 0, 1, shape + 1 ) # +1 for bias
        #print( self.weights )

        
    def UpdateWeights( self, xi, ti ):
        output = self.NetValue( xi ) # deffrent from percetron
        error = ti - output
        #mse = 0.5 * ( error ** 2 )
        
        self.weights[1:] += self.learning_rate * error * xi
        if self.bias is False:         # no bais
            self.weights[0] = 0
        else:
            self.weights[0] += self.learning_rate * error   # bais
        #return mse
    
    def MSE( self, data, target ):
        ## sum( 0.5 * error ** 2 ) / m
        predicted = self.NetValue( data ) # different from perceptron
        error = target - predicted
        mse = 0.5 * ( error ** 2 )
        return mse.mean()
    
    # training model
    def fit( self, x, t ):
        last_mse = 0
        mse = 0
        self.InitializeWeights( x.shape[1] )
        #print( 'w: ', self.weights )
        if self.bias is False:
            self.weights[0] = 100

        for _ in range( self.iters ):
            #print( f'{_}/{self.iters}' )
            for xi, ti in zip( x, t ):
                #mse += self.UpdateWeights( xi, ti ) # sum ( 0.5 * error ** 2 )
                self.UpdateWeights( xi, ti )
                
            
            #mse /= len( x ) # sum( 0.5 * error ** 2 ) / m
            mse = self.MSE( x, t )
            
            print( f'{_}/{self.iters} --> mse: {mse}' )
            #print( last_mse )
            #print( abs( last_mse - mse ) )
            
            if abs( last_mse - mse ) <= self.mse_threshold:
                break
                
            last_mse = mse
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
    