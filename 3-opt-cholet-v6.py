import os
import sys
import pickle
import numpy as np
from itertools import combinations, islice
import multiprocessing
import time


INFINITY = sys.maxsize
data: dict = {}
num_processes = 6#(multiprocessing.cpu_count() // 2)
time_limit = 600

def load_data(folder: str) -> dict:
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  # Convert data to NumPy array


def verify_calculate_weight(solution: np.ndarray, data) -> bool:
    weights = data["weight_Cholet_pb1_bis.pickle"]

    # Ensure solution indices are integers
    solution = solution.astype(int)
    
    node_weights = weights[solution]    

    cumsum_from_left = np.cumsum(node_weights)
    if np.any(cumsum_from_left > 5850):
        return False

    cumsum_from_right = np.cumsum(node_weights[::-1][1:])
    if np.any(cumsum_from_right > 5850):
        return False
    
    return True

def calculate_total_dist(solution: np.ndarray, data) -> int:
    dist_matrix = data["dist_matrix_Cholet_pb1_bis.pickle"]

    # Ensure solution indices are integers
    solution = solution.astype(int)
    
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

def total_distance(solution: np.ndarray, data) -> int:
    if verify_calculate_weight(solution, data):
        return calculate_total_dist(solution, data)
    return INFINITY

def get_all_segments(solution):
    k = 3
    segments = []
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            segments.append(indices)
    return segments


def three_opt_swap1(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j], solution[k:]))
    return new_solution.copy()

def three_opt_swap2(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k][::-1], solution[i:j], solution[k:]))
    return new_solution.copy()

def three_opt_swap3(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j][::-1], solution[k:]))
    return new_solution.copy()

def three_opt_swap4(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k][::-1], solution[i:j][::-1], solution[k:]))
    return new_solution.copy()

def three_opt_swap5(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[i:j][::-1], solution[j:k], solution[k:]))
    return new_solution.copy()

def three_opt_swap6(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[i:j][::-1], solution[j:k][::-1], solution[k:]))
    return new_solution.copy()

def three_opt_swap7(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[i:j], solution[j:k][::-1], solution[k:]))
    return new_solution.copy()


def three_opt_parallel(segments, best_solution, best_distance, start_time, shape, solution_lock, barrier):
    print(f"Process {os.getpid()} started")
    while time.time() - start_time < time_limit:
        with best_distance.get_lock():
            best_distance_for_now = best_distance.value
        new_best_distance = best_distance_for_now
        with solution_lock:
            best_solution_for_now = np.frombuffer(best_solution, dtype='d').reshape(shape)
        new_best_solution = best_solution_for_now.copy()
        for i, j, k in segments:
            for swap in [three_opt_swap1, three_opt_swap2, three_opt_swap3, three_opt_swap4, three_opt_swap5, three_opt_swap6, three_opt_swap7]:
                new_solution = swap(best_solution_for_now, i, j, k)
                new_distance = total_distance(new_solution, data)
                if new_distance < new_best_distance:
                    print(f"Process {os.getpid()} found a better solution: {new_distance}, delta: {best_distance_for_now - new_distance}, time: {time.time() - start_time}")
                    new_best_distance = new_distance
                    new_best_solution = new_solution.copy()
        print(f"Process {os.getpid()} finished one iteration with best distance: {new_best_distance} (delta: {best_distance_for_now - new_best_distance}), time: {time.time() - start_time}")

        # Wait for all processes to finish the iteration
        barrier.wait()

        if new_best_distance < best_distance_for_now:
            with solution_lock and best_distance.get_lock():
                if new_best_distance < best_distance.value:
                        best_distance.value = new_best_distance
                        best_solution[:] = new_best_solution.flatten()
                        print(f"Current best distance: {best_distance.value}, time: {time.time() - start_time}")

        # Wait for all processes to update the best solution
        barrier.wait()

    print(f"Process {os.getpid()} finished")


def four_opt_swap1(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap2(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap3(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k][::-1], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap4(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def four_opt_swap5(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k][::-1], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap6(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k][::-1], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def four_opt_swap7(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k][::-1], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap8(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def four_opt_swap9(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k][::-1], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def three_opt(solution):
    start = time.time()
    best_distance = total_distance(solution, data)
    segments = get_all_segments(solution)
    # np.random.seed(seed)
    np.random.shuffle(segments)
    processes = []

    chunk_size = len(segments) // num_processes
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    if len(segments) % num_processes != 0:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)

    solution_shape = solution.shape
    best_solution = multiprocessing.Array('d', solution.flatten(), lock=False)
    best_solution_lock = multiprocessing.Lock()
    best_distance_value = multiprocessing.Value('d', best_distance)
    barrier = multiprocessing.Barrier(num_processes)

    for chunk in chunks:
        process = multiprocessing.Process(target=three_opt_parallel, args=(chunk, best_solution, best_distance_value, start, solution_shape, best_solution_lock, barrier))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    for process in processes:
        process.terminate()

    best_solution_np = np.frombuffer(best_solution, dtype='d').reshape(solution_shape)
    return best_solution_np, best_distance_value.value


if __name__ == "__main__":


    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]

    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(solution, data))
    best_solution, best_distance = three_opt(solution)
