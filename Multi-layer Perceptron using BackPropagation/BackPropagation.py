import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import layer



class BackPropagation:   
    def __init__(
                self,
                n_inputs = 4,
                n_hiddenlayers = 2,
                hiddenlayersnodes = [5,5],
                n_outputs = 3,
                learning_rate = 0.1,
                n_iters = 1000,
                activation = 'sigmoid',
                bias = False
            ):
        self.activation_function = {
                'sigmoid': ( lambda z: 1 / ( 1 + np.exp( -z ) ) ), 
                'hyperbolicTangent': ( lambda z:  ( np.exp(z) - np.exp(-z) ) / ( np.exp(z) + np.exp(-z) ) )
                #'hyperbolicTangent': ( lambda z: np.tanh( z ) )
            }
        self.deriv_activation_function = {
                'sigmoid': ( lambda z: z * ( 1 - z ) ),
                'hyperbolicTangent': ( lambda z: 1.0 - z**2 ),
            }
        
        self.RunActivation = self.activation_function[ activation ]
        self.RunDerivation = self.deriv_activation_function[ activation ]
        
        self.bias = bias
        self.learning_rate = learning_rate
        self.activation = activation
        self.n_iters = n_iters
        self.layers = []
        
        # Input Layer
        self.layers.append( 
                            layer.Layer( n_inputs, hiddenlayersnodes[0], self.learning_rate, self.RunActivation, self.bias )
                        )
        # Hidden Layers
        for i in range( len( hiddenlayersnodes ) - 1 ):
            self.layers.append( 
                                layer.Layer( hiddenlayersnodes[i], hiddenlayersnodes[i+1], self.learning_rate, self.RunActivation, self.bias )
                            )
        # Output Layer
        self.layers.append( 
                            layer.Layer( hiddenlayersnodes[-1], n_outputs, self.learning_rate, self.RunActivation, self.bias )
                        )
        
    def Forward( self, X ):
        self.layers[0].Forward( X )
        for i in range( 1, len( self.layers ) ):
            self.layers[i].Forward( self.layers[i-1].outputs )
        return self.layers[-1].outputs
    
    def Backward( self, X, y ):
        """ 
            Begin from output layer to update weights backward
        """
        # Output Layers delta and update weights
        out_l2 = self.Forward( X )
       
        error_output = out_l2 - y
        delta = error_output * self.RunDerivation( out_l2 )
        
        layer_inputs = self.layers[-2].outputs
        self.layers[-1].UpdateWeights( layer_inputs, delta  )
            
        # Hidden Layers delta and update weights
        for i in range( 2, len( self.layers ) + 1 ):
            if i == len( self.layers ) : # back to first layer
                layer_inputs = X    
            else: 
                layer_inputs = self.layers[-i-1].outputs
            layer_outputs = self.layers[-i].outputs
            weights = np.transpose( self.layers[-i+1].weights )
            delta = np.matmul( weights, delta ) * self.RunDerivation( layer_outputs )
            self.layers[-i].UpdateWeights( layer_inputs, delta ) 
            
    def Fit( self, data, target ):
        y = self.Encoder( target )
        for _ in range( self.n_iters ):
            error = 0.
            for xi, yi in zip( data, y ):
                self.Backward( xi, yi )
                error += self.Error( xi, yi )
            
            mse = ( 1 / len(data) ) * error
            print( f'Epoch {_}/{self.n_iters } : {mse:.5f}' )
            
    def Encoder( self, target ):
        """
            input: value of ( 0, 1 or 2 )
            output: retrun one hot encoder list like [ 1.2, 2.3, 4.4 ]
        """
        one_hot_encoder = []
        # put breakpoint here
        for target_item in target:
            if( target_item == 0 ): 
                one_hot_encoder.append( np.array( [ 1, 0, 0 ] ) ) # classe 1
            elif( target_item == 1 ):
                one_hot_encoder.append( np.array( [ 0, 1, 0 ] ) ) # classe 2
            elif( target_item == 2 ):
                one_hot_encoder.append( np.array( [ 0, 0, 1 ] ) ) # classe 3        
        return one_hot_encoder
    
    def Predict(self, x):
        #print(self.Forward(x) ) # [0.00534201 0.86741854 0.12015816]
        #print(self.Forward(x).argmax()) # 1
        return self.Forward(x).argmax()
      
    def Error(self, x, y):
        # return error for each pair
        return ( 1 / len(x) ) * sum( np.square( self.Forward(x) - y ) )
    
    def ConfusionMatrix( self, data, target ):
        T00, T11, T22,\
        F01, F02, F10,\
        F12, F20, F21  = 0, 0, 0,\
                         0, 0, 0,\
                         0, 0, 0
        acc = 0
        
        for i, j in zip( target, data ):
            j = self.Predict( j )
            if i == j == 0:
                T00 += 1
            elif i == j == 1:
                T11 += 1
            elif i == j == 2:
                T22 += 1
                
            elif i == 0 and j == 1:
                F01 += 1
            elif i == 0 and j == 2:
                F02 += 1
            elif i == 1 and j == 0:
                F10 += 1
                    
            elif i == 1 and j == 2:
                F12 += 1
            elif i == 2 and j == 0:
                F20 += 1
            elif i == 2 and j == 1:
                F21 += 1
            
            else:
                raise ValueError;
            
        sns.heatmap(
            [ 
                [ T00, F01, F02 ],
                [ F10, T11, F12 ],
                [ F20, F21, T22 ]
            ],
            annot = True
        )
        #plt.show()
        
        acc = T00 + T11 + T22 
        acc /= len( data )
        #acc = ( T00 + T11 + T22 ) / ( F01 + F02 + F10 + F12 + F20 + F21 + T00 + T11 + T22 )
        return acc
    
    
    
"""import sklearn
import sklearn.preprocessing
import sklearn.datasets
import sklearn.model_selection

iris = sklearn.datasets.load_iris()
data = iris.data
labels = iris.target

scaler = sklearn.preprocessing.MinMaxScaler()
data = scaler.fit_transform( data )

labels = np.reshape(labels, (len(labels),1))

x, xx, y, yy = sklearn.model_selection.train_test_split( data, labels, test_size=0.4 )

model = BackPropagation(
                n_inputs = 4,
                n_hiddenlayers = 2,
                hiddenlayersnodes = [5,5],
                n_outputs = 3,
                learning_rate = 0.1,
                n_iters = 1000,
                activation = 'sigmoid',
                bias = False
            )
model.Fit( x, y )

print( model.Predict( xx[0] ), yy[00] )
    
print( model.ConfusionMatrix( xx, yy ) )"""