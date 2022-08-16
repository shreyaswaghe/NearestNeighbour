import numpy as np
import math

arr_type = type(np.array([]))

def euclidean_distance(p1:np.ndarray, p2:np.ndarray)->float:
    '''
    Expect p1, p2 to have order (n_features,1)
    '''
    if type(p1) != arr_type: p1 = np.array(p1)
    if type(p2) != arr_type: p2 = np.array(p2)

    assert p1.shape == p2.shape
    delta_p:np.ndarray = p1-p2

    return math.sqrt(sum(map(lambda val: val**2, delta_p)))


def manhattan_distance(p1:np.ndarray, p2:np.ndarray)->float:
    '''
    Expect p1, p2 to have order (n_features,1)
    '''
    if type(p1) != arr_type: p1 = np.array(p1)
    if type(p2) != arr_type: p2 = np.array(p2)

    assert p1.shape == p2.shape
    delta_p:np.ndarray = p1-p2

    return sum(delta_p)