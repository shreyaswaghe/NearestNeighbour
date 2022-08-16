from nearestneighbour import np, euclidean_distance, arr_type

class knn_classifier:
    """
    K-Nearest Neighbour Classifier, with default k=1.
    
    Can specify heuristic for 'distance'.

    1. Initialise as `knc = knn_classifier([n_neighbors:int?])`.
    2. Then use `knc.fit(X, y)`
    3. Then use `knc.predict(Xi, [heuristic?])`
    """
    def __init__(self, n_neighbors:int=1):
        self.X = None
        self.Y = None
        self.mapping = None
        self.n_neighbors = n_neighbors

        self.n_features = None
        self.n_classes = None
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits model to the data. 
        This method must be called before using `score`/`predict`.

        Args:   
            X (np.ndarray): Expect shape of (n_samples, n_features)
            y (np.ndarray): Expect shape of (n_samples, 1). Elements of y must be hashable.

        Raises:
            Exception: If n_neighbors is set higher than the number of available datapoints.

        Returns:
            self: fluent-designed method
        """
        assert X.shape[0] == y.shape[0] # each sample should have a class
        assert hasattr(y[0], '__hash__') # limitation on data - classes should be hashable

        self.X = X
        self.y = y
        self.n_features = len(X[0])
        self.classes = set(y)
        self.n_classes = len(self.classes)

        if self.n_neighbors > len(y):
            raise Exception("More neighbors than available datapoints, reduce number of neighbors")

        self.mapping = np.array(
            list(zip(X,y,strict=True)), dtype=[('features', np.ndarray), ('class', object)]
        ) # structured np array
        return self


    def predict(self, samples:np.ndarray, heuristic = euclidean_distance):
        """
        Used to predict the class of sample(s), with the specified heuristic.
        When heuristic is not specified, we use the euclidean distance between two feature points.

        Args:
            samples (np.ndarray): These are the test sample points. Expect shape of (n_test_samples, n_features)
            heuristic (heur_func, optional): The function which calculates the 'distance' between twi points. Defaults to euclidean_distance.

        Raises:
            Exception: if samples is not a np.ndarray

        Returns:
            np.ndarray: An array of shape (n_samples, 1), with elements as predicted class of the indexed sample.
        """
        if type(samples) != arr_type:
            raise Exception("Expect samples as a np.ndarray")

        results = []
        for sample in samples:
            self.mapping = sorted(self.mapping, key = lambda item: heuristic(item['features'], sample))
            neighbors_focused = self.mapping[:self.n_neighbors]

            class_votes = dict((klass, 0) for klass in self.classes)
            for neighbor in neighbors_focused:
                class_votes[neighbor['class']]+= 1
            
            results.append(list(class_votes.keys())[
                np.argmax(
                    list(class_votes.values())
                )
            ])
        
        return np.array(results)

    def score(self, samples: np.ndarray, targets: np.ndarray):
        """Calculates the score for the model given sample data and targets.

        Args:
            samples (np.ndarray): These are the test sample points. Expect shape of (n_test_samples, n_features)
            targets (np.ndarray): These are the 'answers' to the predictions the model will make. Expect shape of (n_test_samples, 1)

        Returns:
            float: The percentage of correct answers in this test data. (Greater is better)
        """
        assert samples.shape[0] == targets.shape[0]
        results = self.predict(samples)

        corr,cnt = 0,0
        for e1, e2 in zip(results, targets):
            if e1 == e2: corr += 1
            cnt += 1

        return corr/cnt