'''
This was the first module written in this library.
Deprecated, use knn_classifier instead.

Author: Shreyas Waghe
Date: 07 August, 2022.
'''


from . import np, euclidean_distance

class nearestneighbour_classifier:
    """
    KNeighbour Classifier, with brute search, with k = 1 mandatorily.

    Can specify heuristic for 'distance'.

    1. Initialise as `nnc = nearestneighbour_classifier()`.
    2. Then use `nnc.fit(X, y)`
    3. Then use `nnc.predict(Xi, [heuristic?])`

    In testing against `sklearn.neighbours.KNeighborsClassifier(n_neighbors=1, algorithm='brute')`, 
    this implementation gives times about 10-14 times faster in comparison when used on small datasets.
    This was the first class written in this library.
    """
    def __init__(self):
        self.X = None
        self.y = None
        self.n_features = None
        self.n_classes = None


    def fit(self, X:np.ndarray , y: np.ndarray):
        '''
        Expect X to be the training points (inputs), of shape (n_samples, n_features).

        Expect y to be the classes for points in X, of shape (n_samples, 1).

        Fluent-designed method - Returns self.
        '''
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_classes = len(set(y))

        return self

    def search(self, Xi: np.ndarray, heuristic=None):
        '''
        Expect `Xi` to have dimension (1, n_features).
        Expect `heuristic` to be a function of type (arg: np.ndarray[n_features, 1]) -> Sortable.
        Sortable : Any `dtype` which can be compared - think `int|float|tuple|str`\n

        In absence of `heuristic` argument, uses Euclidean distance between points.

        Follow python3 comparison rules.
        '''
        if(len(Xi) != self.n_features):
            raise Exception(f"Search point has length {self.n_features}, expected number of features {self.n_classes}")
        
        if(heuristic is None):
            heuristic = lambda arg: euclidean_distance(arg, Xi)

        return self.y[
            np.argmin(
                list(map(heuristic, self.X))
            )
        ]

    def predict(self, Xi:np.ndarray, heuristic=None):
        '''
        Alias for nearestneighbour_classifier.search()
        '''
        return self.search(Xi, heuristic)


