import os
import sys
import pickle
import numpy as np
from itertools import combinations
import multiprocessing
import time


INFINITY = sys.maxsize
data: dict = {}
bilat_pairs : dict = {}
num_processes = 16 #(multiprocessing.cpu_count() // 2)
time_limit = 60

def load_data(folder: str) -> dict:
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  # Convert data to NumPy array

def make_dico_bilat_pairs():
    pairs = data["bilat_pairs_Abers_pb2_bis.pickle"]
    for pair in pairs:
        bilat_pairs[pair[0]] = pair[1]
        bilat_pairs[pair[1]] = pair[0]

def permutationList_to_pairList(permutationList: np.ndarray) -> np.ndarray:
    return [(node, node if node not in bilat_pairs.keys() else bilat_pairs[node]) for node in permutationList]

def pairList_to_permutationList(pairList: np.ndarray) -> np.ndarray:
    return [pair[0] for pair in pairList]

def calculate_total_dist(solution: np.ndarray, data) -> int:
    dist_matrix = data["dist_matrix_Abers_pb2_bis.pickle"]

    # Ensure solution indices are integers
    solution = np.array(solution).astype(int)
    
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

def total_distance(solution_paired: np.ndarray, data, best_distance) -> int:
    solution = pairList_to_permutationList(solution_paired)
    distance = calculate_total_dist(solution, data)
    if distance < best_distance:
        return distance
    return INFINITY

def get_all_segments(solution):
    k = 3
    segments = []
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            segments.append(indices)
    return segments

def inverse_node(dist_matrix: dict, solution_paired: np.ndarray, i):
    i_next = int(i+1) if i < len(solution_paired) - 1 else None
    i_prev = int(i-1) if i > 0 else None
    distance = (0 if i_next is None else dist_matrix[int(solution_paired[i][0]), int(solution_paired[i_next][0])]) + (0 if i_prev is None else dist_matrix[int(solution_paired[i][0]), int(solution_paired[i_prev][0])])
    if (distance > (0 if i_next is None else dist_matrix[int(solution_paired[i][1]), int(solution_paired[i_next][0])]) +  (0 if i_prev is None else dist_matrix[int(solution_paired[i][1]), int(solution_paired[i_prev][0])])):
        solution_paired[i] = (solution_paired[i][1], solution_paired[i][0])

def find_best_pairs(solution_paired: np.ndarray, i, j, k):
    dist_matrix = data["dist_matrix_Abers_pb2_bis.pickle"]
    inverse_node(dist_matrix, solution_paired, i)
    inverse_node(dist_matrix, solution_paired, i-1)
    inverse_node(dist_matrix, solution_paired, j)
    inverse_node(dist_matrix, solution_paired, j-1)
    inverse_node(dist_matrix, solution_paired, k)
    inverse_node(dist_matrix, solution_paired, k-1)

def three_opt_swap(solution_paired, i, j, k):
    new_solution_paired = np.concatenate((solution_paired[:i], solution_paired[j:k], solution_paired[i:j], solution_paired[k:]))
    find_best_pairs(new_solution_paired, i, j, k)
    return new_solution_paired.copy()

def three_opt_parallel(segments, best_solution_paired, start_time, shape, solution_lock, event, queue):
    print(f"Process {os.getpid()} started")

    while time.time() - start_time < time_limit:
        print(f"Process {os.getpid()} started iteration")
        best_new_distance = INFINITY
        with solution_lock:
            best_solution_for_now_paired = np.frombuffer(best_solution_paired, dtype='d').reshape(shape)
        best_new_solution_paired = np.array([])
        for i, j, k in segments:
            if time.time() - start_time > time_limit:
                return
            new_solution = three_opt_swap(best_solution_for_now_paired, i, j, k)
            new_distance = total_distance(new_solution, data, best_new_distance)
            if new_distance < best_new_distance:
                best_new_distance = new_distance
                best_new_solution_paired = new_solution
        queue.put((best_new_distance, best_new_solution_paired))

        print(f"Process {os.getpid()} finished iteration")
        event.wait()

def update_solution(best_solution_paired, best_distance, start_time, solution_lock, event, queue):
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
                best_solution_paired[:] = best_queue_solution[1].flatten()
            event.set()
        time.sleep(1)

    event.set()


def three_opt(solution_paired):
    start = time.time()
    best_distance = total_distance(solution_paired, data, INFINITY)
    segments = get_all_segments(solution_paired)
    # np.random.seed(seed)
    # np.random.shuffle(segments)
    processes = []

    chunk_size = len(segments) // num_processes
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    if len(segments) % num_processes != 0:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)

    solution_paired = np.array(solution_paired)
    solution_shape = solution_paired.shape
    best_solution_paired = multiprocessing.Array('d', solution_paired.flatten(), lock=False)
    best_solution_lock = multiprocessing.Lock()
    best_distance_value = multiprocessing.Value('d', best_distance)
    event = multiprocessing.Event()
    queue = multiprocessing.Queue(maxsize=num_processes)

    for chunk in chunks:
        p = multiprocessing.Process(target=three_opt_parallel, args=(chunk, best_solution_paired, start, solution_shape, best_solution_lock, event, queue))
        p.start()
        processes.append(p)

    solution_updater = multiprocessing.Process(target=update_solution, args=(best_solution_paired, best_distance_value, start, best_solution_lock, event, queue))
    solution_updater.start()
    processes.append(solution_updater)

    for p in processes:
        p.join()

    with best_solution_lock:
        solution_found = np.frombuffer(best_solution_paired, dtype='d').reshape(solution_shape)
    with best_distance_value.get_lock():
        best_distance_found = best_distance_value.value

    return solution_found, best_distance_found



if __name__ == "__main__":
    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Abers_2_bis")
    make_dico_bilat_pairs()

    solution = data["init_sol_Abers_pb2_bis.pickle"]
    solution_paired = permutationList_to_pairList(solution)

    print("Initial solution: ", solution)
    print("Initial solution paired: ", solution_paired)
    print("Initial distance: ", total_distance(solution_paired, data, INFINITY))
    print()

    best_solution, best_distance = three_opt(solution_paired)
    print()

    print("Best solution: ", pairList_to_permutationList(best_solution))
    print("Best distance: ", best_distance)


"""
    Number of processes:  16
    CPU Frequency: ~5GHz
    Time limit: 60s
    Best solution: [0.0, 173.0, 77.0, 82.0, 216.0, 143.0, 315.0, 175.0, 319.0, 354.0, 367.0, 353.0, 79.0, 214.0, 174.0, 9.0, 80.0, 318.0, 85.0, 86.0, 320.0, 84.0, 78.0, 352.0, 42.0, 43.0, 40.0, 317.0, 25.0, 152.0, 8.0, 163.0, 176.0, 368.0, 126.0, 122.0, 329.0, 121.0, 120.0, 38.0, 10.0, 11.0, 312.0, 153.0, 172.0, 87.0, 364.0, 24.0, 128.0, 346.0, 129.0, 151.0, 150.0, 132.0, 133.0, 6.0, 7.0, 41.0, 375.0, 88.0, 134.0, 147.0, 164.0, 3.0, 4.0, 5.0, 222.0, 149.0, 73.0, 322.0, 74.0, 209.0, 208.0, 148.0, 130.0, 321.0, 131.0, 336.0, 282.0, 335.0, 283.0, 161.0, 160.0, 340.0, 159.0, 13.0, 14.0, 15.0, 16.0, 258.0, 332.0, 260.0, 328.0, 363.0, 279.0, 280.0, 334.0, 281.0, 156.0, 26.0, 278.0, 23.0, 356.0, 162.0, 39.0, 17.0, 107.0, 348.0, 257.0, 330.0, 256.0, 237.0, 236.0, 139.0, 135.0, 326.0, 136.0, 338.0, 137.0, 35.0, 52.0, 48.0, 327.0, 49.0, 307.0, 50.0, 351.0, 51.0, 113.0, 245.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 289.0, 166.0, 170.0, 242.0, 241.0, 325.0, 239.0, 355.0, 238.0, 349.0, 333.0, 324.0, 102.0, 138.0, 339.0, 337.0, 350.0, 158.0, 344.0, 18.0, 314.0, 36.0, 31.0, 291.0, 32.0, 33.0, 298.0, 299.0, 297.0, 34.0, 60.0, 378.0, 61.0, 377.0, 308.0, 169.0, 168.0, 167.0, 57.0, 296.0, 306.0, 290.0, 304.0, 146.0, 370.0, 46.0, 369.0, 292.0, 195.0, 68.0, 301.0, 69.0, 294.0, 144.0, 347.0, 177.0, 311.0, 310.0, 44.0, 205.0, 376.0, 97.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 19.0, 313.0, 345.0, 331.0, 12.0, 37.0, 101.0, 110.0, 323.0, 233.0, 111.0, 232.0, 171.0, 155.0, 154.0, 316.0, 365.0, 366.0, 157.0, 20.0, 21.0, 22.0, 27.0, 28.0, 362.0, 29.0, 30.0, 140.0, 372.0, 141.0, 373.0, 142.0, 374.0, 178.0, 58.0, 165.0, 47.0, 303.0, 98.0, 361.0, 357.0, 63.0, 359.0, 64.0, 53.0, 358.0, 54.0, 305.0, 360.0, 309.0, 192.0, 55.0, 295.0, 56.0, 371.0, 200.0, 2.0, 302.0, 145.0, 293.0, 1.0, 286.0, 300.0, 204.0, 203.0, 65.0, 287.0, 66.0, 288.0, 285.0, 343.0, 342.0, 284.0, 341.0, 379.0]
    Best distance: 41250.0
"""








