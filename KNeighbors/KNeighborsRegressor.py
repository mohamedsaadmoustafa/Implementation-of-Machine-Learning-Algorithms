import numpy as np
import KNeighbors

class KNeighborsRegressor(KNeighbors):
    def __init__(self, n_neighbors=5, metric='r2', distance_metrix='euclidean', verbose=True):
        KNeighbors.__init__(self, n_neighbors=n_neighbors, metric=metric, verbose=verbose)
        
    def single_prediction(self, test_row):
        k_targets = self.get_k_targets(test_row)
        # return mean of most common
        y_hat = np.mean(k_targets)
        return y_hat

    def score(self, test_data, test_target):
        predict_target = self.predict(test_data)
        evaluate = RegresstionScore(test_target, predict_target)
        return evaluate(metric=self.metric)
    
"""
import sklearn.datasets
x, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 42 )
reg = KNeighborsRegressor(n_neighbors=20)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(reg.score(x_test, y_test))
n_neighbors = reg.best_k(2, x_test, y_test)
print(reg.history)
reg.plot_best_k()
 """;
