import numpy as np

class KNeighbors:
    def __init__(self, n_neighbors=5, metric='F1_Score', distance_metrix='euclidean', verbose=True):  
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.distance_metrix = distance_metrix
        self.best_score = 0 # start with 0 score
            
    def distance(self, row1, row2):
        # calculate euclidean distance for a row pair
        return Distance(row1, row2)(metrix=self.distance_metrix) # minkowski, euclidean or cosine
    
    def fit(self, data, target):
        self.data = data
        self.target = target
        
    def get_k_targets(self, test_row):
        """
            It takes k nearest neighbors whose distances 
            form that point are minimum 
            and computes the average of those values
        """
        # get distances of test_row vs all training rows
        distances = [
            self.distance(test_row, row) 
            for row in self.data
        ]
        
        # get indices of k-nearest neighbors --> k-smallest distances
        k_idx = np.argsort(distances)[:self.n_neighbors]
        # get corresponding y-labels of training data
        k_targets = [self.target[idx] for idx in k_idx]
        return k_targets
    
    def predict(self, test_data):
        # get predictions for every row in test data
        predict_target = [self.single_prediction(test_row) for test_row in test_data]
        return np.array(predict_target)
    
    def best_k(self, range_n_neighbors, test_data, test_target):
        self.history = {
            "n_neighbors": [],
            "test_score": []
        }
        for n_neighbors in range(1, range_n_neighbors+1):
            self.n_neighbors = n_neighbors
            self.fit(self.data, self.target)
            test_score = self.score(test_data, test_target)
            print(f"At n neighbors = {self.n_neighbors} - Score = {test_score} - best_score {self.best_score}")
            self.compare_score(test_score, n_neighbors)
            self.history["n_neighbors"].append(n_neighbors)
            self.history["test_score"].append(test_score)
        return self.best_n_neighbors
    
    def compare_score(self, test_score, n_neighbors):
        if test_score > self.best_score: # look for higher score
            self.best_n_neighbors = int(n_neighbors)
            self.best_score = test_score
    
    def plot_best_k(self):
        plt.plot(self.history["n_neighbors"], self.history["test_score"], alpha=0.5);
        plt.scatter(self.history["n_neighbors"], self.history["test_score"], label="Different Ks");
        plt.scatter(self.best_n_neighbors, self.best_score, c='r', s=100, label="Best n_neighbors")
        plt.ylabel("Score")
        plt.xlabel("n_neighbors")
        plt.legend()
        
