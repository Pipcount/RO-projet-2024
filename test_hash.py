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

if __name__ == '__main__':
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]
    
    start = time.time()
    for _ in range(1000):
        hash(tuple(solution))
    print(time.time() - start)

    start = time.time()
    for _ in range(1000):
        hash(tuple(solution.flatten()))
    print(time.time() - start)
