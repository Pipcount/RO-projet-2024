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
bilat_pairs : dict = {}
num_processes = 24 #(multiprocessing.cpu_count() // 2)
time_limit = 600

# None for no shuffling
# np.random.randint(0, 1000000)
random_seed = 322796
do_screen_clear = True

iteration_without_improvement_threshold = 3
iterations_per_process = 1

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

def make_dico_bilat_pairs():
    pairs = data["bilat_pairs_Abers_pb2.pickle"]
    for pair in pairs:
        bilat_pairs[pair[0]] = pair[1]
        bilat_pairs[pair[1]] = pair[0]

def permutationList_to_pairList(permutationList: np.ndarray) -> np.ndarray:
    return [(node, node if node not in bilat_pairs.keys() else bilat_pairs[node]) for node in permutationList]

def pairList_to_permutationList(pairList: np.ndarray) -> np.ndarray:
    return [pair[0] for pair in pairList]

def save_solution(solution : np.ndarray, best_distance : int, random_seed : int):
    best_distance = int(best_distance)
    os.makedirs(os.path.dirname("./output_data/"), exist_ok=True)
    os.makedirs(os.path.dirname("./output_data/Problem_Abers_2/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./output_data/Problem_Abers_2/{best_distance}_{random_seed}_{time_limit}_{num_processes}/"), exist_ok=True)
    solution_filename = f"output_data/Problem_Abers_2/{best_distance}_{random_seed}_{time_limit}_{num_processes}/solution.csv"
    np.savetxt(solution_filename, solution.flatten(), delimiter=",", fmt="%d", newline=", ")
    params_filename = f"output_data/Problem_Abers_2/{best_distance}_{random_seed}_{time_limit}_{num_processes}/parameters.csv"
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
    dist_matrix = data["dist_matrix_Abers_pb2.pickle"]

    # Ensure solution indices are integers
    solution = np.array(solution).astype(int)
    
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

def total_distance(solution_paired: np.ndarray, data, best_distance, hasheds) -> int:
    solution = pairList_to_permutationList(solution_paired)
    distance = calculate_total_dist(solution, data)
    if distance < best_distance:
        if hash(tuple(solution)) not in hasheds:
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
    dist_matrix = data["dist_matrix_Abers_pb2.pickle"]
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

def three_opt_parallel(segments, best_solution_paired, start_time, shape, solution_lock, event, queue, hashed_solutions, process_number):
    print("\033[90m {}\033[00m" .format(f"Process {os.getpid()} started"))

    while time.time() - start_time < time_limit:
        best_new_distance = INFINITY
        with solution_lock:
            best_solution_for_now_paired = np.frombuffer(best_solution_paired, dtype='d').reshape(shape)
        best_new_solution_paired = np.array([])

        for _ in range(iterations_per_process):
            best_iteration_solution_paired = np.array([])
            best_new_distance = INFINITY
            for i, j, k in segments:
                new_solution = three_opt_swap(best_solution_for_now_paired, i, j, k) if best_new_solution_paired.size == 0 else three_opt_swap(best_new_solution_paired, i, j, k)
                new_distance = total_distance(new_solution, data, best_new_distance, hashed_solutions)

                if new_distance < best_new_distance:
                    best_new_distance = new_distance
                    best_iteration_solution_paired = new_solution
            best_new_solution_paired = best_iteration_solution_paired

        queue[process_number] = (best_new_distance, best_new_solution_paired)

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
                print("\033[92m {}\033[00m" .format("Best solution:"), [int(x) for x in pairList_to_permutationList(best_queue_solution[1])])
                print()
                best_solution_of_all_time = best_queue_solution
                best_solution_iteration = iteration_count
                
            else:
                print("\033[93m {}\033[00m" .format("No new absolute best solution found"))
                print("\033[93m {}\033[00m" .format("Last iteration that improved the best solution:"), best_solution_iteration)
                print("\033[93m {}\033[00m" .format("Best solution found:"), [int(x) for x in pairList_to_permutationList(best_solution_of_all_time[1])])
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

            hashed = hash(tuple(pairList_to_permutationList(best_queue_solution[1])))
            assert hashed not in hashed_solutions, "\033[91m {}\033[00m" .format("Solution already hashed")
            hashed_solutions.append(hashed)

            print_separation()

            event.set()
            event.clear()

    with best_distance.get_lock() and solution_lock:
        best_distance.value = best_solution_of_all_time[0]
        best_solution[:] = best_solution_of_all_time[1].flatten()


def three_opt(solution : np.ndarray):
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
    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Abers_2")
    make_dico_bilat_pairs()

    solution = data["init_sol_Abers_pb2.pickle"]
    solution_paired = permutationList_to_pairList(solution)

    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(np.array(solution_paired), data, INFINITY, []))
    print_separation()

    best_solution, best_distance = three_opt(np.array(solution_paired))
    best_solution = pairList_to_permutationList(best_solution)

    print_separation()
    print("\033[92m {}\033[00m" .format("Best solution:"), [int(x) for x in best_solution])
    print("\033[92m {}\033[00m" .format("Best distance:"), best_distance)
    print_separation()

    save_solution(np.array(best_solution), best_distance, random_seed)


