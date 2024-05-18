import os
import sys
import pickle
import numpy as np
from itertools import combinations, islice
import multiprocessing
import time


INFINITY = sys.maxsize
data: dict = {}
num_processes = 9# (multiprocessing.cpu_count() // 2)
time_limit = 600

def load_data(folder: str) -> dict:
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  # Convert data to NumPy array


def verify_calculate_weight(solution: np.ndarray, data) -> bool:

    weights = data["weight_Cholet_pb1_bis.pickle"]
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

def three_opt_parallel(solution, best_distance, segments, result_queue, start_time, data, better_solution_found, lock):
    for i, j, k in segments:
        if time.time() - start_time > time_limit:
            print(f"Process {os.getpid()} time limit exceeded")
            return
        with lock:
            if better_solution_found.value:
                return
        for swap in [three_opt_swap1, three_opt_swap2, three_opt_swap3, three_opt_swap4]:
            new_solution = swap(solution, i, j, k)
            new_distance = total_distance(new_solution, data)
            if new_distance < best_distance:
                print(f"Process {os.getpid()} found a better solution, distance delta: {best_distance - new_distance}")
                result_queue.put((new_solution, new_distance))
                with lock:
                    better_solution_found.value = True
                return
    

def four_opt_parallel(solution, best_distance, segments, result_queue, better_solution_found, lock, start_time, data):
    for i, j, k in segments:
        for l in range(k + 1, len(solution) - 1):
            if time.time() - start_time > time_limit:
                print(f"Process {os.getpid()} time limit exceeded")
                return
            with lock:
                if better_solution_found.value:
                    return
            for swap in [four_opt_swap1, four_opt_swap2, four_opt_swap3, four_opt_swap4, four_opt_swap5, four_opt_swap6, four_opt_swap7, four_opt_swap8]:
                new_solution = swap(solution, i, j, k, l)
                new_distance = total_distance(new_solution, data)
                if new_distance < best_distance:
                    print(f"Process {os.getpid()} found a better solution, distance delta: {best_distance - new_distance}")
                    result_queue.put((new_solution, new_distance))
                    with lock:
                        better_solution_found.value = True
                    return


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
    improved = True
    best_distance = total_distance(solution, data)


    segments = get_all_segments(solution)
    np.random.shuffle(segments)
    start = time.time()
    processes = []

    while improved and time.time() - start < time_limit:
        improved = False
        chunk_size = len(segments) // num_processes
        chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]

        better_solution_found = multiprocessing.Value('b', False)
        lock = multiprocessing.Lock()
        if len(segments) % num_processes != 0:
            chunks[-2] += chunks[-1]
            chunks.pop(-1)

        result_queue = multiprocessing.Queue()
        for chunk in chunks:
            process = multiprocessing.Process(target=three_opt_parallel, args=(solution, best_distance, chunk, result_queue, start, data, better_solution_found, lock))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        if result_queue.empty() and time.time() - start < time_limit:
            print("!!! No better solution found using 3-opt, trying 4-opt !!!")
            for chunk in chunks:
                process = multiprocessing.Process(target=four_opt_parallel, args=(solution, best_distance, chunk, result_queue, better_solution_found, lock, start, data))
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
        
    for process in processes:
        process.terminate()

    if time.time() - start < time_limit:
        print("Time left: ", time_limit - (time.time() - start))
    else:
        print("Time limit exceeded")

    return solution, best_distance


if __name__ == "__main__":


    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]

    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(solution, data))
    solution, best_distance = three_opt(solution)
    print("Final solution: ", solution)
    print("Final distance: ", best_distance)

