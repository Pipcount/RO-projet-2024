"""
    pip install numpy
    pip install psutil
    pip install pickle
"""

import os
import sys
import pickle
import numpy as np
from itertools import combinations
import multiprocessing
import time
import psutil

INFINITY = sys.maxsize
data: dict = {}
num_processes = 24 #(multiprocessing.cpu_count() // 2)
time_limit = 600

# None for no shuffling
# np.random.randint(0, 1000000)
random_seed = 322796
do_screen_clear = True

iteration_without_improvement_threshold = 3
iterations_per_process = 5

def clear_screen():
    print("\033c", end="")

def print_separation():
    print()
    print(os.get_terminal_size()[0] * "=")
    print()

def print_new_iteration(iteration_count):
    if(do_screen_clear):
        print_separation()
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

def verify_calculate_weight(solution : np.ndarray, data : dict) -> bool:
    weights = data["bonus_multi_commodity_Cholet_pb1_bis.pickle"]
    return verify_calculate_weight_indexed(solution, weights, 0) and verify_calculate_weight_indexed(solution, weights, 1)

def verify_calculate_weight_indexed(solution : np.ndarray, weights : np.ndarray, index : int) -> bool:
    # Ensure solution indices are integers
    solution = solution.astype(int)
    
    node_weights = [weight[index] for weight in weights[solution]]

    cumsum_from_left = np.cumsum(node_weights)
    if np.any(cumsum_from_left > 5850):
        return False

    cumsum_from_right = np.cumsum(node_weights[::-1][1:])
    if np.any(cumsum_from_right > 5850):
        return False
    
    return True

def save_solution(solution : np.ndarray, best_distance : int, random_seed : int):
    best_distance = int(best_distance)
    os.makedirs(os.path.dirname("./output_data/"), exist_ok=True)
    os.makedirs(os.path.dirname("./output_data/Problem_Cholet_1_bis_bonus/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./output_data/Problem_Cholet_1_bis_bonus/{best_distance}_{random_seed}_{time_limit}_{num_processes}/"), exist_ok=True)
    solution_filename = f"output_data/Problem_Cholet_1_bis_bonus/{best_distance}_{random_seed}_{time_limit}_{num_processes}/solution.csv"
    np.savetxt(solution_filename, solution.flatten(), delimiter=",", fmt="%d", newline=", ")
    params_filename = f"output_data/Problem_Cholet_1_bis_bonus/{best_distance}_{random_seed}_{time_limit}_{num_processes}/parameters.csv"
    with open(params_filename, 'w') as f:
        f.write(f"Best Distance found, {best_distance}\n")
        f.write(f"Random Seed, {random_seed}\n")
        f.write(f"Time Limit, {time_limit}\n")
        f.write(f"Number of Processes, {num_processes}\n")
        f.write(f"CPU frequency, {int(psutil.cpu_freq().current)} MHz\n") 
        f.write(f"Iteration without improvement threshold, {iteration_without_improvement_threshold}\n") 
        f.write(f"Iterations per process, {iterations_per_process}\n")
        f.close()
    print("\033[92m {}\033[00m" .format("Solution saved"))

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

        for _ in range(iterations_per_process):
            best_iteration_solution = np.array([])
            best_new_distance = INFINITY
            for i, j, k in segments:
                new_solution = three_opt_swap(best_solution_for_now, i, j, k) if best_new_solution.size == 0 else three_opt_swap(best_new_solution, i, j, k)
                new_distance = total_distance(new_solution, data, best_new_distance, hashed_solutions)

                if new_distance < best_new_distance:
                    best_new_distance = new_distance
                    best_iteration_solution = new_solution
            best_new_solution = best_iteration_solution

        queue[process_number] = (best_new_distance, best_new_solution)

        event.wait()


def update_solution(best_solution, best_distance, start_time, solution_lock, event, queue, hashed_solutions):
    print("\033[90m {}\033[00m" .format("Solution updater started"))
    best_solutions_found = []
    best_solution_of_all_time = (INFINITY, None)
    iteration_count = 0
    best_solution_iteration = 0
    iteration_without_improvement = 0
    while time.time() - start_time < time_limit:
        if(None not in queue):
            iteration_count += 1
            if do_screen_clear:
                clear_screen()
            print_new_iteration(iteration_count)
            print_separation()
            best_queue_solution = (INFINITY, None)

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
                best_solution_iteration = iteration_count
                
            else:
                print("\033[93m {}\033[00m" .format("No new absolute best solution found"))
                print("\033[93m {}\033[00m" .format("Last iteration that improved the best solution:"), best_solution_iteration)
                print("\033[93m {}\033[00m" .format("Best solution found:"), [int(x) for x in best_solution_of_all_time[1]])
                print("\033[93m {}\033[00m" .format("With distance:"), best_solution_of_all_time[0])
                print()

            with best_distance.get_lock() and solution_lock:

                iteration_variation = best_distance.value - best_queue_solution[0]
                if iteration_variation == 0:
                    iteration_without_improvement += 1
                else:
                    if iteration_variation > 0:
                        best_solutions_found.append(best_queue_solution)
                    iteration_without_improvement = 0

                print("\033[94m {}\033[00m" .format(f"Previous iteration best distance:"), best_distance.value)
                print("\033[94m {}\033[00m" .format(f"Iteration best distance:"), best_queue_solution[0])
                print("\033[94m {}\033[00m" .format(f"Iteration variation value:"), iteration_variation)
                print("\033[94m {}\033[00m" .format(f"Time:"), time.time() - start_time)

                if iteration_without_improvement >= iteration_without_improvement_threshold:
                    if(len(best_solutions_found) > 1): 
                        print()
                        best_solutions_found.pop(-1)
                        print("\033[93m{}\033[00m" .format(f"Going back to a previous solution whit a distance of {best_solutions_found[-1][0]} in order to find a new way to get a better solution"))
                        iteration_without_improvement = 0
                    else:
                        print("\033[91m {}\033[00m" .format("No new way to get a better solution found"))
                        print_separation()
                        best_distance.value = best_solution_of_all_time[0]
                        best_solution[:] = best_solution_of_all_time[1].flatten()
                        return
                    
                best_distance.value = best_solutions_found[-1][0]
                best_solution[:] = best_solutions_found[-1][1].flatten()

            hashed = hash(tuple(best_queue_solution[1]))
            assert hashed not in hashed_solutions, "\033[91m {}\033[00m" .format("Solution already hashed")
            hashed_solutions.append(hashed)

            print_separation()

            event.set()
            event.clear()

    with best_distance.get_lock() and solution_lock:
        best_distance.value = best_solution_of_all_time[0]
        best_solution[:] = best_solution_of_all_time[1].flatten()


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
    best_solution_lock = multiprocessing.Lock()
    best_distance_value = multiprocessing.Value('d', best_distance)
    event = multiprocessing.Event()
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
    weights = data["bonus_multi_commodity_Cholet_pb1_bis.pickle"]
    weights[-1] = (-5850, -5850)
    weights[-2] = (-5850, -5850)

    print("Initial solution:", solution)
    print("Initial distance:", total_distance(solution, data, INFINITY, []))
    print_separation()

    best_solution, best_distance = three_opt(solution)

    print_separation()
    print("\033[92m {}\033[00m" .format("Best solution:"), [int(x) for x in best_solution])
    print("\033[92m {}\033[00m" .format("Best distance:"), best_distance)
    print_separation()

    save_solution(best_solution, best_distance, random_seed)


"""
    Number of processes: 24
    Random-seed: 322796
    CPU: ~5GHz
    Time limit: 600s
    Best solution: [0, 100, 101, 184, 115, 153, 102, 149, 148, 3, 208, 202, 58, 59, 60, 62, 63, 54, 194, 196, 197, 6, 7, 8, 218, 198, 119, 193, 176, 64, 96, 5, 93, 121, 128, 127, 189, 15, 94, 23, 25, 26, 27, 28, 24, 126, 125, 124, 123, 95, 103, 104, 105, 106, 136, 135, 134, 107, 16, 40, 122, 133, 17, 18, 19, 108, 169, 172, 168, 167, 43, 44, 45, 46, 47, 48, 49, 166, 34, 132, 131, 130, 35, 36, 37, 38, 39, 165, 29, 116, 67, 11, 186, 10, 1, 2, 199, 164, 163, 221, 229, 147, 4, 12, 187, 68, 191, 146, 117, 69, 158, 118, 159, 70, 144, 143, 41, 13, 14, 195, 142, 227, 120, 222, 42, 175, 20, 138, 55, 56, 139, 57, 9, 145, 51, 52, 53, 65, 129, 137, 192, 61, 185, 141, 140, 50, 21, 200, 22, 31, 32, 33, 71, 173, 201, 174, 188, 73, 150, 190, 30, 181, 72, 66, 74, 205, 97, 209, 217, 215, 231, 114, 179, 180, 110, 111, 92, 157, 171, 228, 170, 161, 183, 220, 182, 91, 206, 230, 79, 162, 98, 99, 203, 154, 207, 80, 81, 82, 83, 84, 210, 85, 86, 87, 88, 78, 89, 90, 212, 219, 213, 214, 216, 204, 211, 155, 177, 178, 109, 77, 160, 224, 113, 76, 112, 152, 151, 226, 75, 156, 225, 223, 232]
    Best distance: 40009.0
"""

"""
    Number of processes: 24
    Random-seed: 319216
    CPU: ~5GHz
    Time limit: 600s
    Best solution: [0, 100, 101, 184, 115, 79, 162, 98, 99, 203, 154, 153, 102, 148, 202, 9, 145, 51, 52, 53, 65, 138, 55, 56, 139, 57, 129, 58, 59, 61, 185, 120, 222, 42, 175, 144, 143, 41, 13, 14, 142, 227, 195, 62, 63, 54, 196, 194, 197, 6, 7, 193, 8, 218, 198, 119, 176, 64, 96, 5, 93, 121, 189, 128, 94, 23, 25, 26, 27, 28, 24, 126, 125, 124, 123, 95, 103, 104, 105, 106, 136, 135, 134, 107, 16, 40, 122, 133, 17, 18, 19, 108, 169, 172, 168, 167, 43, 44, 45, 46, 47, 48, 49, 166, 34, 132, 131, 130, 35, 36, 37, 38, 39, 165, 29, 1, 2, 199, 116, 67, 186, 10, 11, 117, 69, 163, 221, 229, 164, 147, 4, 12, 68, 191, 146, 187, 158, 15, 127, 118, 159, 70, 20, 137, 60, 192, 141, 140, 50, 21, 200, 22, 31, 32, 33, 71, 72, 66, 74, 205, 75, 152, 151, 226, 97, 112, 156, 225, 209, 217, 215, 231, 114, 149, 173, 201, 174, 188, 73, 150, 190, 30, 181, 3, 208, 220, 110, 92, 157, 111, 180, 179, 171, 228, 170, 161, 183, 182, 91, 206, 230, 207, 80, 81, 82, 83, 84, 210, 85, 86, 87, 88, 78, 89, 90, 212, 219, 213, 214, 216, 204, 211, 155, 177, 178, 109, 77, 160, 224, 113, 76, 223, 232]
    Best distance: 41964.0
"""