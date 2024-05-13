import os
import sys
import pickle
import signal
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

    cumsum_from_left = np.cumsum(node_weights)
    if np.any(cumsum_from_left > 5850):
        return False

    cumsum_from_right = np.cumsum(node_weights[::-1][1:])
    if np.any(cumsum_from_right > 5850):
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
    k = 4
    segments = []
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            segments.append(indices)
    return segments

def four_opt_swap(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_parallel(solution, best_distance, segments, result_queue):
    for i, j, k, l in segments:
        new_solution = four_opt_parallel(solution, i, j, k, l)
        new_distance = total_distance(new_solution)
        if new_distance < best_distance:
            result_queue.put((new_solution, new_distance))
            return

def four_opt(solution):
    improved = True
    best_distance = total_distance(solution)

    def handler(signum, frame):
        raise TimeoutError("Time limit exceeded")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(600) # 10 minutes timeout

    segments = get_all_segments(solution)


    try:
        while improved:
            improved = False
            chunk_size = len(segments) // num_processes
            chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
            result_queue = multiprocessing.Queue()
            processes = []
            for chunk in chunks:
                process = multiprocessing.Process(target=four_opt_parallel, args=(solution, best_distance, chunk, result_queue))
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

    test = get_all_segments(solution)
    # print("Initial solution: ", solution)
    # print("Initial distance: ", total_distance(solution))
    # solution, best_distance = four_opt(solution)
    # print("Final solution: ", solution)
    # print("Final distance: ", best_distance)
