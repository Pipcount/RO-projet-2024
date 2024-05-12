import os
import sys
import pickle
import signal
import cProfile
import pstats
import numpy as np
from itertools import combinations


INFINITY = sys.maxsize
data: dict = {}

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
    segments = []
    for i in range(1, len(solution) - 2):
        for j in range(i + 1, len(solution) - 1):
            segments.append((i, j))
    return segments

def three_opt_swap(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j], solution[k:]))
    return new_solution.copy()


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
            segments = get_all_segments_k(solution, 2)
            for i, j in segments:
                for k in range(j + 1, len(solution)):
                    new_solution = three_opt_swap(solution, i, j, k)
                    new_distance = total_distance(new_solution)
                    if new_distance < best_distance:
                        solution = new_solution
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    # print("Improved distance: ", best_distance)
                    # print("Solution: ", solution)
                    break
    except TimeoutError:
        print("Time limit exceeded")

    signal.alarm(0)
    return solution, best_distance


def get_all_segments_k(solution, k):
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            yield indices


def k_opt_swap(solution, indices):
    segments = [solution[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
    new_solution = np.concatenate((solution[:indices[0]], *segments[::-1], solution[indices[-1]:]))
    return new_solution.copy()

def k_opt(solution, k):
    improved = True
    best_distance = total_distance(solution)

    def handler(signum, frame):
        raise TimeoutError("Time limit exceeded")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(600)  # 10 minutes timeout

    try:
        while improved:
            improved = False
            for segment in get_all_segments_k(solution, k):
                new_solution = k_opt_swap(solution, segment)
                new_distance = total_distance(new_solution)
                if new_distance < best_distance:
                    solution = new_solution
                    best_distance = new_distance
                    improved = True
                    break
            # if improved:
            #     print("Improved distance: ", best_distance)
            #     print("Solution: ", solution)
    except TimeoutError:
        print("Time limit exceeded")

    signal.alarm(0)
    return solution, best_distance


if __name__ == "__main__":

    # Profiler:
    profiler = cProfile.Profile()
    profiler.enable()

    load_data("input_data/Probleme_Cholet_1_bis")
    initial_solution = data["init_sol_Cholet_pb1_bis.pickle"]
    improved_solution, improved_distance = three_opt(initial_solution)
    print("three_opt improved solution: ", improved_solution)
    print("three_opt improved distance: ", improved_distance)
    # improved_solution, improved_distance = k_opt(initial_solution, 3)
    # print("k_opt improved solution: ", improved_solution)
    # print("k_opt improved distance: ", improved_distance)

    # Stop profiling
    profiler.disable()
    # Save profile results to a file
    profiler.dump_stats("profile_results.prof")

    # Alternatively, you can print the profile results to the console
    # profiler.print_stats()

    # If you want to analyze the profile results further, you can use pstats
    stats = pstats.Stats("profile_results.prof")
    stats.strip_dirs()
    stats.sort_stats("cumulative")  # Choose sorting method (e.g., "cumulative", "time", "calls")
    stats.print_stats()  # Print the profile statistics
