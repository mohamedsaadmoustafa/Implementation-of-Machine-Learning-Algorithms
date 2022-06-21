import numpy as np

class Distance:
    """
        Distance Metrics
    """
    def __init__(self, x, y, axis=0, p_value=1):
        #assert(len(x)==len(y))
        self.x = x
        self.y = y
        self.p_value = p_value
        self.axis = axis

    def pairwise_dist(self):
        """
            Args:
                x: N x D numpy array
                y: M x D numpy array
            Return:
                    dist: N x M array, where dist2[i, j] is the euclidean distance between 
                    x[i, :] and y[j, :]

            To calculate Minkowski distance set p_value == 1
            To calculate Euclidean distance set p_value == 2
        """
        z = np.abs(self.y - self.x)** self.p_value
        z = np.sum(z, axis=self.axis)
        return z**(1/self.p_value)
    
    def Cosine(self):
        """
            This distance metric is used mainly to calculate similarity between two vectors. 
            It is measured by the cosine of the angle between two vectors and determines 
            whether two vectors are pointing in the same direction. 
            It is often used to measure document similarity in text analysis.
            With KNN: distance gives us a new perspective to a business problem 
            and lets us find some hidden information in the data 
            which we didn’t see using the above two distance matrices. 
        """
        a = np.linalg.norm(self.x) * np.linalg.norm(self.y)
        b = np.dot(self.x, self.y)
        return a / b
    
    def __call__(self, metrix):
        metrix = metrix.lower()
        if metrix == 'euclidean'  : 
            self.p_value = 2
            return self.pairwise_dist()
        elif metrix == 'minkowski': return self.pairwise_dist()
        elif metrix == 'cosine'   : return self.Cosine()
        else: raise ValueError
            
            
#r1 = np.linspace(1, 10)
#r2 = np.linspace(10, 100)
#Distance(r1, r2)(metrix='minkowski')# minkowski euclidean cosine
