import numpy as np

class Evaluate:
    """
        Evaluation metrics are a measure of how good a model performs 
        and how well it approximates the relationship. 
    """
    def __init__(self, actual, predicted):
        assert(actual.shape == predicted.shape)
        self.actual = actual
        self.predicted = predicted
        
        
        
class RegresstionScore(Evaluate):
    """
        Evaluation metrics are a measure of how good a model performs 
        and how well it approximates the relationship. 
        like: MSE, MAE, R-squared, and RMSE.
    """
    def __init__(self, actual, predicted):
        super().__init__(actual, predicted)
        # Residual = actual value — predicted value
        self.Residual = np.subtract(
                self.actual, 
                self.predicted
            ) 
        # mean value of a sample
        self.actual_bar = np.sum(self.actual) / len(self.actual)
        
    def mse(self):
        # Mean Squared Error (MSE)
        # MSE penalizes large errors
        #MSE = (np.linalg.norm(self.actual - self.predicted)**2) / len(self.actual)
        return np.square( 
                self.Residual
            ).mean()
    
    def mae(self):
        # Mean Absolute Error (MAE)
        return np.absolute( 
                self.Residual
            ).mean()

    def rmse(self):
        # Root Mean Squared Error (RMSE)
        return np.sqrt(self.mse())
    
    def r2(self):
        """
        R-squared (R2) is a statistical measure
        represents the proportion of the variance for a dependent variable 
        that's explained by an independent variable or variables in a regression model.
        """
        # residual sum of squares (RSS)
        RSS = np.sum(
            np.square( 
                self.Residual
            )
        )
        # Total sum of squares
        TSS = np.sum(
            np.square( 
                np.subtract(
                    self.actual, 
                    self.actual_bar
                ) 
            )
        )
        return 1 - ( RSS / TSS )
    
    def __call__(self, metric):
        metrics = {
            "mse":  self.mse(),
            "mae":  self.mae(),
            "rmse": self.rmse(),
            "r2":   self.r2(),
        }
        try: 
            return metrics[metric]
        except:
            return metrics
          
          
          
class ClassificationScore(Evaluate):
    """
        Accuracy, Precision, and Recall
    """
    def __init__(self, actual, predicted):
        super().__init__(actual, predicted)
        # True Positive
        self.TP = ((self.actual == 1) & (self.predicted == 1)).sum()
        # True Negative
        self.TN = ((self.actual == 0) & (self.predicted == 0)).sum() 
        # False Positive
        self.FP = ((self.actual == 1) & (self.predicted == 0)).sum()
        # False Negative
        self.FN = ((self.actual == 0) & (self.predicted == 1)).sum()
    
    def Accuracy(self):
        """
            Accuracy is a valid choice of evaluation for classification problems 
            which are well balanced and not skewed or No class imbalance.
            
            Accuracy = (TP+TN)/(TP+FP+FN+TN)
        """
        return (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)
    
    def Precision(self):
        """
            Precision is a valid choice of evaluation metric 
            when we want to be very sure of our prediction. 
            For example: 
                    If we are building a system to predict 
                    if we should decrease the credit limit on a particular account, 
                    we want to be very sure about our prediction 
                    or it may result in customer dissatisfaction.
            
            Precision = (TP)/(TP+FP)
        """
        return (self.TP) / (self.TP + self.FP)

    
    def Recall(self):
        """
            Recall is a valid choice of evaluation metric 
            when we want to capture as many positives as possible. 
            For example: 
                    If we are building a system to predict 
                    if a person has cancer or not, 
                    we want to capture the disease even if we are not very sure.
                    
            Recall = (TP)/(TP+FN)
        """
        return (self.TP) / (self.TP + self.FN)

    def F1_Score(self):
        """
            The F1 score is a number between 0 and 1 and is the harmonic mean of precision and recall.
            When to use? 
                We want to have a model with both good precision and recall.
            F1 score sort of maintains a balance between the precision and recall for your classifier. 
                * If your precision is low, the F1 is low
                * If the recall is low again your F1 score is low.
            
            f1 = 2*((Precision * Recall)/(Precision + Recall))
        """
        return 2*((self.Precision() * self.Recall()) / (self.Precision() + self.Recall()))
    
    
    def BinaryCrossEntropy(self):
        """
            Log Loss / Binary Crossentropy
            Log Loss takes into account the uncertainty of your prediction based on how much it varies from the actual label.
            -(y * log(p) + (1-y) * log(1-p))
        """
        predicted = np.clip(self.predicted, 1e-7, 1 - 1e-7)        

        # Calculating loss
        loss_positive = np.dot(self.actual.T, np.log(predicted)) # y * log(p)
        loss_negative = np.dot((1 - self.actual).T, np.log(1 - predicted)) # (1-y) * log(1-p))
        return -np.mean(loss_positive + loss_negative)#, axis=0)
    
    def Categorical_Crossentropy(self):
        """
            When the output of a classifier is multiclass prediction probabilities. 
            We generally use Categorical Crossentropy in case of Neural Nets. 
            In general, minimizing Categorical cross-entropy gives greater accuracy for the classifier.
            
            Where y_pred is a matrix of probabilities with 
                shape = (n_samples, n_classes) 
                y_true is an array of class labels
            
            log_loss(y_true, y_pred, eps=1e-15)
        """
        predicted = np.clip(self.predicted, 1e-15, 1 - 1e-15)  # eps=1e-15    
        return - np.sum(np.dot(self.actual.T, np.log(predicted)))
    
    def AUC(self):

        """
            AUC is the area under the ROC curve.
            AUC ROC indicates how well the probabilities 
            from the positive classes are separated from the negative classes
            
            Sensitivity: The probability that the model predicts a positive outcome 
                for an observation when indeed the outcome is positive. 
                This is also called the “true positive rate.”
                Sensitivty = Recall = TPR(True Positive Rate) = TP/(TP+FN)
            
            Specificity: The probability that the model predicts a negative outcome 
                for an observation when indeed the outcome is negative. 
                This is also called the “true negative rate.”
                1- Specificity = FPR(False Positive Rate)= FP/(TN+FP)
                
            The closer the AUC is to 1, the better the model.
        """
        True_Positive_Rate = self.Recall()
        False_Positive_Rate = self.FP / (self.TN + self.FP) # false alarm rate 
        return True_Positive_Rate, False_Positive_Rate

    def __call__(self, metric):
        metrics = {
            "Accuracy":  self.Accuracy(),
            "Precision":  self.Precision(),
            "Recall":  self.Recall(),
            "F1_Score":  self.F1_Score(),
            "AUC":   self.AUC(),
        }
        if len(np.unique(self.actual)) == 2: 
            metrics["Crossentropy"] =  self.BinaryCrossEntropy()
        else: 
            metrics["Crossentropy"] = self.Categorical_Crossentropy()
            
        try: 
            return metrics[metric]
        except:
            return metrics
