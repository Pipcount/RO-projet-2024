import os
import sys
import pickle
import numpy as np
from itertools import combinations
import time


INFINITY = sys.maxsize
data: dict = {}



def load_data(folder: str) -> dict:
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  # Convert data to NumPy array

def get_all_segments(solution):
    k = 3
    segments = []
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            segments.append(indices)
    return segments

def three_opt_swap(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j], solution[k:]))
    return new_solution.copy()

def three_opt_swap_v2(solution, i, j, k):
    new_solution = np.empty_like(solution)
    new_solution[:i] = solution[:i]
    new_solution[i:i + (k - j)] = solution[j:k]
    new_solution[i + (k - j):i + (k - j) + (j - i)] = solution[i:j]
    new_solution[i + (k - j) + (j - i):] = solution[k:]
    return new_solution

def four_opt_swap(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k], solution[i:j], solution[l:]))
    return new_solution.copy()

def k_opt_swap(solution, indices):
    segments = [solution[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
    new_solution = np.concatenate((solution[:indices[0]], *segments[::-1], solution[indices[-1]:]))
    return new_solution.copy()

def k_opt_swap_v2(solution, indices):
    segments = [solution[indices[i]:indices[i+1]] for i in range(len(indices)-1)]

    reversed_segments = [np.flip(segment) for segment in segments]

    new_solution = np.concatenate((solution[:indices[0]], *reversed_segments, solution[indices[-1]:]))
    return new_solution.copy()



if __name__ == '__main__':
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]

    # Testing 3-opt swap
    segments = get_all_segments(solution)
    start = time.time()
    for i, j, k in segments:
        new_solution = three_opt_swap(solution, i, j, k)
    print("Time taken for 3-opt swap: ", time.time() - start)

    # start = time.time()
    # for i, j, k in segments:
    #     new_solution = three_opt_swap_v2(solution, i, j, k)
    # print("Time taken for 3-opt swap v2: ", time.time() - start)

    # solution = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # segments = [(1, 4, 7)]
    # for i, j, k in segments:
    #     new_solution = three_opt_swap(solution, i, j, k)
    #     print(solution)
    #     print(new_solution)
    
    # segments = get_all_segments(solution)
    # 
    # # three_opt_swap_solution = []
    # start = time.time()
    # for i, j, k in segments:
    #     new_solution = three_opt_swap(solution, i, j, k)
    #     # three_opt_swap_solution.append(new_solution)
    #
    # print("Time taken for 3-opt swap: ", time.time() - start)
    #
    # # k_opt_swap_solution = []
    # start = time.time()
    # for indices in segments:
    #     new_solution = k_opt_swap(solution, indices)
    #     # k_opt_swap_solution.append(new_solution)
    # print("Time taken for k-opt swap: ", time.time() - start)
    #
    # start = time.time()
    # for indices in segments:
    #     new_solution = k_opt_swap_v2(solution, indices)
    # print("Time taken for k-opt swap v2: ", time.time() - start)
    #
    # # for i in range(len(three_opt_swap_solution)):
    # #     assert np.array_equal(three_opt_swap_solution[i], k_opt_swap_solution[i])
    #
    #
    # # 4-opt
    # segments = [ (1, 4, 7, 10), (2, 5, 8, 11), (3, 6, 9, 12) ]
    # start = time.time()
    # for e in range(100000):
    #     for i, j, k, l in segments:
    #         new_solution = four_opt_swap(solution, i, j, k, l)
    # print("Time taken for 4-opt swap: ", time.time() - start)
    #
    # start = time.time()
    # for e in range(100000):
    #     for indices in segments:
    #         new_solution = k_opt_swap(solution, indices)
    # print("Time taken for k-opt swap: ", time.time() - start)



    # comparing 3-opt and 4-opt
    # three_opt_segments = [ (1, 4, 7), (2, 5, 8), (3, 6, 9) ]
    # four_opt_segments = [ (1, 4, 7, 10), (2, 5, 8, 11), (3, 6, 9, 12) ]
    # start = time.time()
    # for e in range(1000000):
    #     for i, j, k in three_opt_segments:
    #         new_solution = three_opt_swap(solution, i, j, k)
    # print("Time taken for 3-opt swap: ", time.time() - start)

    # start = time.time()
    # for e in range(1000000):
    #     for i, j, k, l in four_opt_segments:
    #         new_solution = four_opt_swap(solution, i, j, k, l)
    # print("Time taken for 4-opt swap: ", time.time() - start)

