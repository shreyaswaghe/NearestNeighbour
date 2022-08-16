 
# Nearest Neighbour
This is a simple package that provides for functionality of a brute-force K-Nearest-Neighbour classifier and regressor . 

## Run Locally  

Initial setup 

~~~ 
  cd <your-project-path>
  git clone https://github.com/shreyaswaghe/NearestNeighbour 
  pip install -r requirements.txt
~~~

To use
~~~Python3
from nearestneighbour.knn_regressor import knn_regressor
from nearestneighbour.knn_classifier import knn_classifier

knn = knn_classifier/knn_regressor(n_neighbors=3)
knn.fit(X, y)
knn.predict(X)
knn.score(X,y)
~~~
More detailed information in docstrings in package modules.
See demo.py for a working example.

## License  

[MIT](https://choosealicense.com/licenses/mit/)
 
## Lessons Learned  
I learnt a lot about the underlying mathematics and mechanics of KNN algorithms and translating abstract concepts into code.

This was my first time building a project from scratch using Numpy, and though the learning curve was the most time-consuming and difficult aspect of writing this package, I gained a lot of confidence in using this important Python library.



 
## Acknowledgements  
- Introduction to Machine Learning with Python, O'Reilly
- ReadMe Editor (in making this ReadMe), SumitNalavade
