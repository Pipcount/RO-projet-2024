import numpy as np
import sys, os, pickle
import time

INFINITY = sys.maxsize
data: dict = {}

def load_data(folder: str) -> dict:
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  # Convert data to NumPy array

def verify_calculate_weight_leftnright(solution: np.ndarray) -> bool:
    weights = data["weight_Cholet_pb1_bis.pickle"]
    node_weights = weights[solution]

    first_empty_index = np.where(node_weights == -5850)[0][0]

    cumsum_left = np.cumsum(node_weights[:first_empty_index])
    if np.any(cumsum_left > 5850):
        return False

    cumsum_right = np.cumsum(node_weights[first_empty_index + 1:-1])
    if np.any(cumsum_right > 5850):
        return False

    return True

def verify_calculate_weight_classic(solution: list) -> bool:
    weight = 0
    weights = data["weight_Cholet_pb1_bis.pickle"]  
    for node in solution:
        weight += weights[node]
        if weight < 0:
            weight = 0
        if weight > 5850:
            return False
    return True


def verify_calculate_weight_new(solution: list) -> bool:
    weights = data["weight_Cholet_pb1_bis.pickle"]
    node_weights = weights[solution]    

    cumsum_from_left = np.cumsum(node_weights)
    if np.any(cumsum_from_left > 5850):
        return False

    cumsum_from_right = np.cumsum(np.flip(node_weights)[1:])
    if np.any(cumsum_from_right > 5850):
        return False
    
    return True

def verify_calculate_weight_new_v2(solution: list) -> bool:
    weights = data["weight_Cholet_pb1_bis.pickle"]
    node_weights = weights[solution]    

    cumsum_from_left = np.cumsum(node_weights)
    if np.any(cumsum_from_left > 5850):
        return False

    cumsum_from_right = np.cumsum(node_weights[::-1][1:])
    if np.any(cumsum_from_right > 5850):
        return False
    
    return True

if __name__ == '__main__':
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]
    
    start = time.time()
    for i in range(10000):
        verify_calculate_weight_leftnright(solution)
    print("Time taken for leftnright: ", time.time() - start)

    start = time.time()
    for i in range(10000):
        verify_calculate_weight_classic(solution)
    print("Time taken for classic: ", time.time() - start)

    start = time.time()
    for i in range(10000):
        verify_calculate_weight_new(solution)
    print("Time taken for new: ", time.time() - start)


    start = time.time()
    for i in range(10000):
        verify_calculate_weight_new_v2(solution)
    print("Time taken for new v2: ", time.time() - start)


