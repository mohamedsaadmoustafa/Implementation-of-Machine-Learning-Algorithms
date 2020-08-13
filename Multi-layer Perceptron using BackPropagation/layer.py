import numpy as np

class Layer:
    def __init__(self, inputs_shape, outputs_shape, learning_rate, RunActivation, bias_bool):
        self.learning_rate = learning_rate
        self.RunActivation = RunActivation
        self.inputs_shape = inputs_shape
        self.outputs_shape = outputs_shape
        self.weights = []
        self.bias = []
        self.bias_bool = bias_bool
        self.outputs = []
        for i in range( outputs_shape ):
            self.weights.append( [ np.random.randn() for _ in range( inputs_shape ) ] )
            if bias_bool:
                self.bias.append( np.random.randn() )
            else:
                self.bias.append( 0 )
                            
    def Forward( self, inputs ):
        outputs = []
        for i in range( self.outputs_shape ):
            # multiply weights by inputs
            outputs.append( sum( np.multiply( self.weights[i], inputs ) ) )
        # add bias and apply activation function
        if self.bias_bool:
            self.outputs = np.add( outputs, self.bias )
        else:
            self.outputs = np.add( outputs, 0)
        self.outputs = self.RunActivation( self.outputs )
        return outputs
    
    def UpdateWeights( self, inputs, delta ):
        for i in range( self.outputs_shape ):
            for j in range( len( self.weights[i] ) ):
                self.weights[i][j] -= self.learning_rate * delta[i] * inputs[j]                
                if self.bias_bool: self.bias[i] -= self.learning_rate * delta[i]