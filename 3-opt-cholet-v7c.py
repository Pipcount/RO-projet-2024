import os
import sys
import pickle
import numpy as np
from itertools import combinations
import multiprocessing
import time

INFINITY = sys.maxsize
data: dict = {}
num_processes = 24 #(multiprocessing.cpu_count() // 2)
time_limit = 30

# None for no shuffling
# np.random.randint(0, 1000000)
random_seed = None
random_behavior = True

def print_separation():
    print()
    print(os.get_terminal_size()[0] * "=")
    print()

def print_new_iteration(iteration_count):
    text : str = f" Iteration {iteration_count} "
    separators : str = ((os.get_terminal_size()[0] - len(text)) // 2) * "-"
    full_text : str = separators + text + separators
    if len(full_text) < os.get_terminal_size()[0]:
        full_text = full_text + "-"
    print("\033[93m" + full_text + "\033[00m")

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

def total_distance(solution: np.ndarray, data, best_distance, hasheds) -> int:
    distance = calculate_total_dist(solution, data)
    if distance < best_distance:
        if hash(tuple(solution)) not in hasheds:
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


def three_opt_swap(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j], solution[k:]))
    return new_solution

def safe_get_hashed_solutions(hashed_solutions):
    items = []
    start = time.time()
    while True:
        try:
            item = hashed_solutions.get_nowait()
            items.append(item)
        except:
            break
        if time.time() - start > 2:
            print("get timeout")
            break
    return items

def three_opt_parallel(segments, best_solution, start_time, shape, solution_lock, event, queue, hashed_solutions, process_number):
    print("\033[90m {}\033[00m" .format(f"Process {os.getpid()} started"))

    while time.time() - start_time < time_limit:
        best_new_distance = INFINITY
        with solution_lock:
            best_solution_for_now = np.frombuffer(best_solution, dtype='d').reshape(shape)
        best_new_solution = np.array([])
        for i, j, k in segments:

            new_solution = three_opt_swap(best_solution_for_now, i, j, k)
            
            new_distance = total_distance(new_solution, data, best_new_distance, hashed_solutions)

            if new_distance < best_new_distance:
                best_new_distance = new_distance
                best_new_solution = new_solution

        queue[process_number] = (best_new_distance, best_new_solution)

        if not random_behavior:
            event.wait()


def update_solution(best_solution, best_distance, start_time, solution_lock, event, queue, hashed_solutions):
    print("\033[90m {}\033[00m" .format("Solution updater started"))
    best_solution_of_all_time = (INFINITY, None)
    iteration_count = 0
    while time.time() - start_time < time_limit:
        if(None not in queue):
            iteration_count += 1
            print_new_iteration(iteration_count)
            print_separation()
            if not random_behavior:
                print("All processes finished iteration, updating best solution")
            else:
                print("Updating best solution")
            print()
            best_queue_solution = (INFINITY, None)

            if(random_behavior):
                random_range = np.random.permutation(num_processes)
                for i in random_range:
                    queue_solution = queue[i]
                    if queue_solution[0] < best_queue_solution[0]:
                        best_queue_solution = queue_solution
            else:
                for i in range(num_processes):
                    queue_solution = queue[i]
                    if queue_solution[0] < best_queue_solution[0]:
                        best_queue_solution = queue_solution
            for i in range(num_processes):
                queue[i] = None

            if best_queue_solution[0] < best_solution_of_all_time[0]:
                print("\033[92m {}\033[00m" .format("New absolute best solution found with a distance of"), best_queue_solution[0])
                print("\033[92m {}\033[00m" .format("Best solution:"), [int(x) for x in best_queue_solution[1]])
                print()
                best_solution_of_all_time = best_queue_solution
            else:
                print("\033[93m {}\033[00m" .format("No new absolute best solution found"))
                print()

            with best_distance.get_lock() and solution_lock:
                print("\033[94m {}\033[00m" .format(f"Iteration best distance:"), best_queue_solution[0])
                print("\033[94m {}\033[00m" .format(f"Iteration variation value:"), best_distance.value - best_queue_solution[0])
                print("\033[94m {}\033[00m" .format(f"Time:"), time.time() - start_time)
                best_distance.value = best_queue_solution[0]
                best_solution[:] = best_queue_solution[1].flatten()

            if not random_behavior:
                assert hash(tuple(best_queue_solution[1])) not in hashed_solutions, "\033[91m {}\033[00m" .format("Solution already hashed")
            hashed_solutions.append(hash(tuple(best_queue_solution[1])))

            if not random_behavior:
                print()
                print("All processes can now begin a new iteration")
            print_separation()

            if not random_behavior:
                event.set()
                event.clear()

    event.set()


def three_opt(solution):
    start = time.time()
    best_distance = total_distance(solution, data, INFINITY, [])
    segments = get_all_segments(solution)

    if random_seed is not None:
        np.random.seed(random_seed)
        np.random.shuffle(segments)

    processes = []

    chunk_size = len(segments) // num_processes
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    if len(segments) % num_processes != 0:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)

    solution_shape = solution.shape
    best_solution = multiprocessing.Array('d', solution.flatten(), lock=False)
    print(best_solution[:])
    best_solution_lock = multiprocessing.Lock()
    best_distance_value = multiprocessing.Value('d', best_distance)
    event = multiprocessing.Event()
    # queue = multiprocessing.Queue(maxsize=num_processes)
    hashed_solutions = multiprocessing.Manager().list()
    queue = multiprocessing.Manager().list()
    for _ in range(num_processes):
        queue.append(None)

    process_number = 0
    for chunk in chunks:
        p = multiprocessing.Process(target=three_opt_parallel, args=(chunk, best_solution, start, solution_shape, best_solution_lock, event, queue, hashed_solutions, process_number))
        p.start()
        processes.append(p)
        process_number += 1

    solution_updater = multiprocessing.Process(target=update_solution, args=(best_solution, best_distance_value, start, best_solution_lock, event, queue, hashed_solutions))
    solution_updater.start()
    processes.append(solution_updater)

    solution_updater.join()

    for p in processes:
        p.terminate()

    with best_solution_lock:
        solution_found = np.frombuffer(best_solution, dtype='d').reshape(solution_shape)
    with best_distance_value.get_lock():
        best_distance_found = best_distance_value.value
    
    print("Random seed:", random_seed)

    return solution_found, best_distance_found



if __name__ == "__main__":
    print_separation()
    print("Number of processes:", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]
    print("Initial solution:", solution)
    print("Initial distance:", total_distance(solution, data, INFINITY, []))
    print_separation()

    best_solution, best_distance = three_opt(solution)

    print_separation()
    print("\033[92m {}\033[00m" .format("Best solution:"), [int(x) for x in best_solution])
    print("\033[92m {}\033[00m" .format("Best distance:"), best_distance)
    print_separation()


"""
    Number of processes: 16
    CPU Frequency: ~5GHz
    Time limit: 600s
    Best solution: [0.0, 80.0, 81.0, 82.0, 83.0, 84.0, 210.0, 85.0, 86.0, 87.0, 88.0, 78.0, 89.0, 90.0, 212.0, 219.0, 213.0, 214.0, 216.0, 204.0, 211.0, 155.0, 177.0, 178.0, 109.0, 169.0, 169.0, 168.0, 16.0, 40.0, 40.0, 19.0, 18.0, 133.0, 167.0, 166.0, 165.0, 29.0, 1.0, 2.0, 199.0, 116.0, 67.0, 186.0, 10.0, 11.0, 117.0, 69.0, 163.0, 221.0, 229.0, 93.0, 121.0, 128.0, 127.0, 189.0, 15.0, 94.0, 23.0, 26.0, 27.0, 28.0, 24.0, 25.0, 126.0, 125.0, 124.0, 123.0, 95.0, 103.0, 104.0, 105.0, 34.0, 132.0, 131.0, 130.0, 37.0, 38.0, 39.0, 106.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 107.0, 16.0, 17.0, 122.0, 122.0, 133.0, 17.0, 18.0, 19.0, 108.0, 172.0, 172.0, 168.0, 167.0, 136.0, 135.0, 134.0, 166.0, 34.0, 132.0, 131.0, 130.0, 35.0, 36.0, 37.0, 38.0, 39.0, 164.0, 147.0, 4.0, 12.0, 68.0, 191.0, 146.0, 187.0, 158.0, 118.0, 70.0, 58.0, 59.0, 195.0, 62.0, 63.0, 54.0, 194.0, 196.0, 197.0, 6.0, 7.0, 193.0, 8.0, 218.0, 198.0, 119.0, 176.0, 64.0, 96.0, 5.0, 159.0, 144.0, 143.0, 41.0, 13.0, 222.0, 14.0, 142.0, 227.0, 120.0, 42.0, 175.0, 20.0, 21.0, 22.0, 31.0, 32.0, 33.0, 71.0, 72.0, 66.0, 207.0, 112.0, 231.0, 114.0, 149.0, 173.0, 201.0, 174.0, 188.0, 73.0, 150.0, 190.0, 30.0, 181.0, 148.0, 202.0, 9.0, 145.0, 51.0, 52.0, 53.0, 65.0, 55.0, 56.0, 139.0, 57.0, 138.0, 129.0, 137.0, 60.0, 192.0, 61.0, 185.0, 141.0, 140.0, 50.0, 200.0, 3.0, 208.0, 220.0, 110.0, 92.0, 157.0, 111.0, 180.0, 179.0, 171.0, 228.0, 170.0, 161.0, 183.0, 182.0, 203.0, 154.0, 160.0, 224.0, 113.0, 76.0, 223.0, 232.0]
    Best distance: 38804.0
"""

"""
    Number of processes: 24
    CPU Frequency: ~5GHz
    Time limit: 600s
    Best solution: [0, 77, 100, 101, 184, 115, 137, 60, 185, 141, 192, 140, 50, 9, 145, 51, 52, 53, 65, 138, 55, 56, 139, 57, 129, 175, 96, 5, 5, 93, 121, 189, 128, 127, 15, 94, 23, 24, 26, 27, 28, 25, 126, 125, 124, 123, 95, 229, 147, 4, 186, 10, 1, 2, 199, 105, 106, 107, 40, 19, 16, 17, 18, 19, 108, 169, 169, 172, 168, 167, 34, 132, 131, 130, 37, 38, 39, 106, 136, 135, 134, 136, 135, 134, 107, 108, 172, 168, 16, 40, 122, 122, 133, 17, 18, 133, 167, 43, 44, 45, 46, 47, 48, 49, 43, 44, 45, 46, 47, 48, 49, 166, 34, 132, 131, 130, 35, 36, 37, 38, 39, 165, 164, 221, 221, 229, 104, 29, 116, 67, 11, 12, 187, 68, 191, 146, 117, 69, 163, 103, 158, 23, 118, 159, 143, 41, 13, 222, 14, 142, 120, 42, 200, 3, 208, 33, 71, 72, 66, 207, 112, 231, 114, 149, 173, 201, 174, 188, 73, 150, 190, 30, 181, 148, 202, 58, 59, 61, 227, 195, 62, 63, 54, 194, 197, 6, 7, 193, 8, 218, 198, 119, 176, 64, 144, 196, 197, 6, 7, 193, 8, 218, 198, 119, 176, 64, 70, 20, 21, 22, 31, 32, 220, 110, 92, 157, 111, 180, 179, 171, 228, 170, 161, 183, 182, 203, 154, 160, 224, 113, 76, 223, 232]
    Best distance: 31363.0
"""