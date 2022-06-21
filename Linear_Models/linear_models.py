class linear_models:
    def __init__( 
        self, 
        dtype="float64", learning_rate=0.01,
        iters=300, normalize=False, 
        copy_X=True, method='normal', alpha=0.1, batch_size=32, tolerance=1e-07,
        is_shuffle=True, random_state=42, metric='mse', optimizer='SGD'
    ):
        """FUTURE NOTE: add all attrs at config file"""#########################
        #print('Model created')
        self.dtype = dtype
        # Converting learning_rate to NumPy array with selected dtype
        self.learning_rate = learning_rate
        self.iters = int(iters)
        self.normalize = normalize
        self.copy_X = copy_X
        self.method = method
        self.alpha = alpha 
        self.batch_size = batch_size
        self.metric = metric
        self.random_state = random_state
        self.optimizer = optimizer
        # Converting tolerance to NumPy array with selected dtype
        self.tolerance = tolerance # stopping criteria
        self.is_shuffle = is_shuffle
        # Run to check parameters
        self.check_parameters()

    def check_parameters(self):
        """
             To check parameters values entered by user
        """
        if self.iters < 0:
            raise ValueError("'n_iter' must be greater than zero")
        if self.learning_rate <= 0:
            raise ValueError("'learning_rate' must be greater than zero")
        if self.tolerance <= 0:
            raise ValueError("'tolerance' must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("'batch_size' must be greater than zero")
        if self.alpha <= 0:
            raise ValueError("'alpha' must be greater than zero")
        #if self.method not in ['solver', 'sgd', 'batchgd', 'svd', 'normal', 'qrsolver']:
        #    raise ValueError("'method' must be one of solver, sgd, batchgd, svd, normal, qrsolver ")
    
    def shuffle(self, data, target):
        p = np.random.permutation(len(data)) # axis=0 by default
        # p = np.random.shuffle(np.arange(len(data)))
        return data[p], target[p]
            
    def preprocessing(self, data_, target_):
        assert( len(data_) == len(target_) )
        # save data to a copy
        if self.copy_X: data, target = data_.copy(), target_.copy()
        else: data, target = data_, target_
        # normalize data 
        if self.normalize: data = self.normalize_2d(data)
        # shuffle data and target
        if self.is_shuffle: 
            data, target = self.shuffle(data, target)
        return data, target

    def normalize_2d(self, matrix):
        """
            Normalize data
            x -> (x - mu(x)) / std(x)
        """
        # Only this is changed to use 2-norm put 2 instead of 1
        # normalized matrix
        #return matrix / np.linalg.norm(matrix, 1)  
        return (matrix - np.mean(matrix, 0)) / np.std(matrix, 0)
    
    #def __del__(self):
        # Deleting (Calling destructor)
        # del model
       # print('Destructor called, Model deleted.')
