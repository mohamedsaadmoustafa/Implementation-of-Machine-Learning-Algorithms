import numpy as np
import KNeighbors

class KNeighborsClassifier(KNeighbors):
    def __init__(self, n_neighbors=5, metric='F1_Score', distance_metrix='euclidean', verbose=True):
        KNeighbors.__init__(self, n_neighbors=n_neighbors, metric=metric, verbose=verbose)
    
    def single_prediction(self, test_row):
        # get corresponding y-labels of training data
        k_labels = self.get_k_targets(test_row)
        # return most common label
        return np.argmax(np.bincount(k_labels))

    def score(self, test_data, test_target):
        predict_target = self.predict(test_data)
        evaluate = ClassificationScore(test_target, predict_target)
        return evaluate(metric=self.metric)
    
    
n_clusters = 5
n_features = 10

x, y = sklearn.datasets.make_blobs(
    n_samples=1000, n_features=n_features,
    centers=n_clusters, cluster_std=1.1,
    center_box=(-10.0, 10.0), shuffle=True, 
    random_state=None, return_centers=False
)

print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 42 )
print(f"x shape: {x_train.shape}, {x_test.shape}")
print(f"y shape: {y_train.shape}, {y_test.shape}")

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(clf.score(x_test, y_test))
#k = clf.best_k(25, x_test, y_test)
#clf.plot_best_k()
