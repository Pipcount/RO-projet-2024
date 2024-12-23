import os
import sys
import pickle
import numpy as np
from itertools import combinations
import multiprocessing
import time


INFINITY = sys.maxsize
data: dict = {}
num_processes = 6 #(multiprocessing.cpu_count() // 2)
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

def total_distance(solution: np.ndarray, data, best_distance) -> int:
    distance = calculate_total_dist(solution, data)
    if distance < best_distance:
        if verify_calculate_weight(solution, data):
            return distance
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
    return new_solution

def three_opt_swap2(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k][::-1], solution[i:j], solution[k:]))
    return new_solution

def three_opt_swap3(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j][::-1], solution[k:]))
    return new_solution

def three_opt_swap4(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k][::-1], solution[i:j][::-1], solution[k:]))
    return new_solution

def three_opt_swap5(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[i:j][::-1], solution[j:k], solution[k:]))
    return new_solution

def three_opt_swap6(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[i:j][::-1], solution[j:k][::-1], solution[k:]))
    return new_solution

def three_opt_swap7(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[i:j], solution[j:k][::-1], solution[k:]))
    return new_solution

def three_opt_parallel(segments, best_solution, start_time, shape, solution_lock, event, queue):
    print(f"Process {os.getpid()} started")

    while time.time() - start_time < time_limit:
        print(f"Process {os.getpid()} started iteration")
        best_new_distance = INFINITY
        with solution_lock:
            best_solution_for_now = np.frombuffer(best_solution, dtype='d').reshape(shape)
        best_new_solution = np.array([])
        for i, j, k in segments:
            for swap in [three_opt_swap1]: #, three_opt_swap2, three_opt_swap3, three_opt_swap4, three_opt_swap5, three_opt_swap6, three_opt_swap7]:
                if time.time() - start_time > time_limit:
                    return
                new_solution = swap(best_solution_for_now, i, j, k)
                new_distance = total_distance(new_solution, data, best_new_distance)
                if new_distance < best_new_distance:
                    best_new_distance = new_distance
                    best_new_solution = new_solution
        queue.put((best_new_distance, best_new_solution))

        print(f"Process {os.getpid()} finished iteration")
        event.wait()


def update_solution(best_solution, best_distance, start_time, solution_lock, event, queue):
    print("Solution updater started")
    while time.time() - start_time < time_limit:
        event.clear()
        if queue.qsize() == num_processes:
            print("All processes finished iteration, updating best solution")
            best_queue_solution = (INFINITY, None)
            for _ in range(num_processes):
                distance, solution = queue.get()
                if distance < best_queue_solution[0]:
                    best_queue_solution = (distance, solution)
            with best_distance.get_lock() and solution_lock:
                print(f"Current best distance: {best_queue_solution[0]}, delta: {best_distance.value - best_queue_solution[0]}, time: {time.time() - start_time}")
                best_distance.value = best_queue_solution[0]
                best_solution[:] = best_queue_solution[1].flatten()
            event.set()
        time.sleep(1)

    event.set()


def three_opt(solution):
    start = time.time()
    best_distance = total_distance(solution, data, INFINITY)
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
    event = multiprocessing.Event()
    queue = multiprocessing.Queue(maxsize=num_processes)

    for chunk in chunks:
        p = multiprocessing.Process(target=three_opt_parallel, args=(chunk, best_solution, start, solution_shape, best_solution_lock, event, queue))
        p.start()
        processes.append(p)

    solution_updater = multiprocessing.Process(target=update_solution, args=(best_solution, best_distance_value, start, best_solution_lock, event, queue))
    solution_updater.start()
    processes.append(solution_updater)

    for p in processes:
        p.join()

    with best_solution_lock:
        solution_found = np.frombuffer(best_solution, dtype='d').reshape(solution_shape)
    with best_distance_value.get_lock():
        best_distance_found = best_distance_value.value

    return solution_found, best_distance_found



if __name__ == "__main__":

    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]
    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(solution, data, INFINITY))
    best_solution, best_distance = three_opt(solution)
    print("Best solution: ", best_solution)
    print("Best distance: ", best_distance)


"""
3-opt-swap_1:
distance: 42044

    solution = np.array([  0,  80,  81,  82,  83,  84, 210,  85,  86,  87,  88,  78,  89,  90,
 212, 219, 213, 214, 216, 204, 211, 155, 177, 169, 168, 167,  43,  44,
  45,  46,  47,  48,  49, 166,  34, 132, 131, 130,  35,  36,  37,  38,
  39, 165, 164, 147,   4,  12,  68, 191, 146, 187, 158,  26,  27,  28,
  24,  25, 126, 125, 124, 123,  95, 221, 229,  29,   1,   2, 199, 116,
  67, 186,  10,  11, 117,  69, 163, 103, 104, 105, 106, 136, 135, 134,
 107,  16,  40, 122, 133,  17,  18,  19, 108, 172,  93, 121, 189, 128,
 127,  15,  94,  23, 118,  70,  58,  59,  60, 195,  62,  63,  54, 196,
 194, 197,   6,   7,   8, 218, 198, 119, 193, 176,  64,  96,   5, 159,
 144, 143,  41,  13,  14, 142, 227, 120, 222,  42, 175,  20,  55,  56,
 139,  57, 138,   9, 145,  51,  52,  53,  65, 129, 137,  61, 185, 141,
 192, 140,  50,  21, 200,  22,  31, 202,  32,  33,  71,  72,  66,  74,
 205,  75, 152, 151, 226,  97, 112, 156, 209, 217, 215, 231, 114, 149,
 173, 201, 174, 188,  73, 150, 190,  30, 181, 148,   3, 208, 220, 110,
  92, 157, 111, 180, 179, 171, 228, 170, 161, 183, 182, 178, 109,  77,
 100, 101, 184, 115,  79, 162,  98,  99, 203,  91, 230, 206, 154, 153,
 102, 207, 160, 224, 113,  76, 225, 223, 232])


3-opt-swap_2:
distance 49226

3-opt-swap_3:
distance: 49110

3-opt-swap_4:
distance: 49403

3-opt-swap_5:
distance: 49403


3-opt-swap_6:
distance: 49403

3-opt-swap_7:
distance: 49403


"""








