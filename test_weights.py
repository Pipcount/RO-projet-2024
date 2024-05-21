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

def verify_calculate_weight_new_v3(solution: list) -> bool:
    weights = data["weight_Cholet_pb1_bis.pickle"]
    node_weights = weights[solution]    
    left_pointer = 0
    right_pointer = len(node_weights) - 2 # -2 because the last element is -5850
    left_sum = 0
    right_sum = 0
    while left_pointer < right_pointer:
        left_sum += node_weights[left_pointer]
        right_sum += node_weights[right_pointer]
        if left_sum > 5850 or right_sum > 5850:
            return False
        left_pointer += 1
        right_pointer -= 1
    return True

def calculate_total_dist(solution: np.ndarray) -> int:
    dist_matrix = data["dist_matrix_Cholet_pb1_bis.pickle"]
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

if __name__ == '__main__':
"""
    Currently the fastest function is verify_calculate_weight_new_v2

"""




    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]
    
    start = time.time()
    for i in range(10000):
        np.random.shuffle(solution)
        verify_calculate_weight_new_v2(solution)
    print("Time taken for new v2: ", time.time() - start)

    start = time.time()
    for i in range(10000):
        np.random.shuffle(solution)
        calculate_total_dist(solution)
    print("Time taken for total distance: ", time.time() - start)

    start = time.time()
    for i in range(10000):
        np.random.shuffle(solution)
        verify_calculate_weight_new_v3(solution)
    print("Time taken for new v3: ", time.time() - start)
    # start = time.time()
    # for i in range(10000):
    #     verify_calculate_weight_leftnright(solution)
    # print("Time taken for leftnright: ", time.time() - start)
    #
    # start = time.time()
    # for i in range(10000):
    #     verify_calculate_weight_classic(solution)
    # print("Time taken for classic: ", time.time() - start)
    #
    # start = time.time()
    # for i in range(10000):
    #     verify_calculate_weight_new(solution)
    # print("Time taken for new: ", time.time() - start)




