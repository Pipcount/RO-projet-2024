import os
import sys
import pickle
import signal
import cProfile
import pstats
import numpy as np
from itertools import combinations
import multiprocessing


INFINITY = sys.maxsize
data: dict = {}
num_processes = multiprocessing.cpu_count() // 2

def load_data(folder: str) -> dict:
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  # Convert data to NumPy array

def verify_calculate_weight(solution: np.ndarray) -> bool:
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

def calculate_total_dist(solution: np.ndarray) -> int:
    dist_matrix = data["dist_matrix_Cholet_pb1_bis.pickle"]
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

def total_distance(solution: np.ndarray) -> int:
    if verify_calculate_weight(solution):
        return calculate_total_dist(solution)
    return INFINITY

def get_all_segments(solution):
    k = 3
    segments = []
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            segments.append(indices)
    return segments

def three_opt_swap(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j], solution[k:]))
    return new_solution.copy()

def three_opt_parallel(solution, best_distance, segments, result_queue):
    for i, j, k in segments:
        new_solution = three_opt_swap(solution, i, j, k)
        new_distance = total_distance(new_solution)
        if new_distance < best_distance:
            result_queue.put((new_solution, new_distance))
            return

def three_opt(solution):
    improved = True
    best_distance = total_distance(solution)

    def handler(signum, frame):
        raise TimeoutError("Time limit exceeded")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(600) # 10 minutes timeout


    try:
        while improved:
            improved = False
            segments = get_all_segments(solution)
            chunk_size = len(segments) // num_processes
            chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
            result_queue = multiprocessing.Queue()
            processes = []
            for chunk in chunks:
                process = multiprocessing.Process(target=three_opt_parallel, args=(solution, best_distance, chunk, result_queue))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
            
            while not result_queue.empty():
                new_solution, new_distance = result_queue.get()
                if new_distance < best_distance:
                    solution = new_solution
                    best_distance = new_distance
                    improved = True
            print("Improved distance: ", best_distance)
            print("Solution: ", solution)

    except TimeoutError:
        print("Time limit exceeded")

    print("Time left: ", signal.alarm(0))
    return solution, best_distance

if __name__ == "__main__":
    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]

    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(solution))
    solution, best_distance = three_opt(solution)
    print("Final solution: ", solution)
    print("Final distance: ", best_distance)


"""
local minimum found with 3-opt in around 7 minutes:

    [0,  77, 100, 101, 184, 115,  79,  98,  99,  91, 206, 230, 162, 177, 178, 109,  80,  81, 82,  83,  84, 210,  85,  86,  87,  88,  78, 214,  89, 213,  90, 212, 219, 216, 204, 211, 155,  93,  15, 189, 121, 128, 127,  94,  23,  26,  27,  28,  24,  25, 126, 125, 124, 123, 95, 221, 229,  29,   1,   2, 199, 116,  67, 186,  10,  11,  12, 187 , 68, 191, 146, 117, 69, 163, 103, 104, 105,  34, 132, 131, 130 , 35,  36,  37,  38,  39, 106, 136, 135, 134, 107,  16,  40, 122, 133,  17,  18,  19, 108, 169, 172, 168, 167 , 43 , 44 , 45 , 46 , 47, 48,  49, 166, 165, 164, 147,   4, 158, 118, 144, 143,  41,  13, 222,  14, 142, 227, 120, 42, 175,  58,  59,  60, 192 , 61, 185, 141, 140,  50,   9, 145 , 51,  52,  53,  65,  55, 56, 139,  57, 138, 129, 137, 195,  62,  63,  54, 194, 196, 197,   6,   7, 193,   8, 218, 198, 119, 176,  64,  96,   5, 159,  70,  20,  21, 200,  22,  31, 202 , 32,  72,  66, 207, 160, 224, 113,  76, 156, 225, 209, 217, 215, 231, 114, 149,   3, 208,  33,  71, 173, 201, 174, 188,  73, 150, 190,  30, 181, 148, 220, 110,  92, 157, 111, 180, 179, 171, 228, 170, 161, 183, 182, 203, 154, 153, 102,  74, 205,  75, 152, 151, 226,  97, 112, 223, 232]

distance: 42439

"""
