import os
import sys
import pickle
import signal
import cProfile
import pstats
import numpy as np
from itertools import combinations, islice
import multiprocessing

INFINITY = sys.maxsize
data: dict = {}
bilat_pairs : dict = {}
num_processes = 10 # multiprocessing.cpu_count() // 2

def load_data(folder: str):
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

def calculate_total_dist(solution_paired: np.ndarray) -> int:
    dist_matrix = data["dist_matrix_Abers_pb2.pickle"]
    solution = pairList_to_permutationList(solution_paired)
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

def get_all_segments(solution: np.ndarray) -> list:
    k = 3
    segments = []
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            segments.append(indices)
    return segments

def inverse_node(dist_matrix: dict, solution_paired: np.ndarray, i):
    distance = (0 if (i == len(solution_paired) - 1) else dist_matrix[solution_paired[i][0], solution_paired[i+1][0]]) + (0 if (i == 1) else dist_matrix[solution_paired[i][0], solution_paired[i-1][0]])
    if (distance > (0 if (i == len(solution_paired) - 1) else dist_matrix[solution_paired[i][1], solution_paired[i+1][0]]) +  (0 if (i == 1) else dist_matrix[solution_paired[i][1], solution_paired[i-1][0]])):
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

def three_opt_parallel(solution_paired, best_distance, segments, result_queue):
    for i, j, k in segments:
        new_solution_paired = three_opt_swap(solution_paired, i, j, k)
        new_distance = calculate_total_dist(new_solution_paired)
        if new_distance < best_distance:
            result_queue.put((new_solution_paired, new_distance))
            return

def three_opt(solution_paired):
    improved = True
    best_distance = calculate_total_dist(solution_paired)

    def handler(signum, frame):
        raise TimeoutError("Time limit exceeded")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(600) # 10 minutes timeout

    segments = get_all_segments(solution_paired)

    try:
        while improved:
            improved = False
            chunk_size = len(segments) // num_processes
            print("Chunk size: ", chunk_size)
            chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
            result_queue = multiprocessing.Queue()
            processes = []
            for chunk in chunks:
                process = multiprocessing.Process(target=three_opt_parallel, args=(solution_paired, best_distance, chunk, result_queue))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
            
            while not result_queue.empty():
                new_solution, new_distance = result_queue.get()
                if new_distance < best_distance:
                    solution_paired = new_solution
                    best_distance = new_distance
                    improved = True
            print("Improved distance: ", best_distance)
            print("Solution: ", pairList_to_permutationList(solution_paired))

    except TimeoutError:
        print("Time limit exceeded")

    print("Time left: ", signal.alarm(0))
    return solution_paired, best_distance

if __name__ == "__main__":
    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Abers_2")
    make_dico_bilat_pairs()

    solution = data["init_sol_Abers_pb2.pickle"]
    solution_paired = permutationList_to_pairList(solution)

    print("Initial solution: ", solution)
    print("Initial solution paired: ", solution_paired)
    print("Initial distance: ", calculate_total_dist(solution_paired))

    print()

    solution_paired, best_distance = three_opt(solution_paired)
    print("Final solution: ", pairList_to_permutationList(solution_paired))
    print("Final distance: ", best_distance)

"""
    With 10 processes:
        Time left:  415/600
        Final solution:  [0, 181, 305, 34, 272, 35, 47, 151, 155, 232, 330, 331, 231, 275, 71, 279, 72, 303, 73, 25, 161, 109, 283, 298, 306, 230, 83, 280, 307, 54, 1, 31, 32, 299, 346, 26, 27, 229, 300, 228, 302, 24, 28, 293, 152, 150, 147, 309, 88, 273, 127, 106, 13, 295, 14, 15, 177, 103, 323, 104, 288, 320, 157, 285, 36, 194, 308, 193, 20, 21, 107, 199, 62, 63, 64, 105, 135, 322, 312, 266, 252, 
        339, 336, 260, 48, 49, 66, 78, 262, 255, 55, 156, 2, 139, 33, 319, 163, 162, 251, 141, 140, 3, 4, 5, 9, 301, 10, 11, 128, 345, 12, 80, 144, 143, 142, 6, 7, 8, 256, 324, 257, 258, 337, 259, 43, 287, 44, 45, 46, 84, 304, 85, 297, 86, 146, 145, 160, 311, 310, 253, 261, 133, 334, 134, 254, 269, 263, 333, 164, 189, 74, 75, 76, 329, 340, 124, 265, 264, 111, 321, 108, 318, 112, 113, 207, 296, 206, 
        130, 131, 132, 276, 284, 235, 342, 29, 30, 154, 343, 341, 153, 267, 338, 268, 18, 19, 137, 70, 122, 99, 115, 271, 57, 98, 125, 126, 204, 40, 96, 97, 335, 328, 136, 114, 56, 278, 68, 179, 53, 159, 281, 138, 59, 332, 60, 212, 61, 282, 222, 79, 205, 291, 327, 326, 203, 315, 202, 313, 201, 274, 241, 233, 286, 316, 101, 314, 102, 270, 158, 277, 37, 38, 39, 41, 42, 16, 290, 17, 149, 148, 22, 58, 289, 292, 23, 294, 317, 325, 50, 51, 344, 52, 347]
        Final distance:  41451
"""