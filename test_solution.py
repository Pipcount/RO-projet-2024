import numpy as np
import os
import pickle

def calculate_total_dist(solution: np.ndarray, data) -> int:
    dist_matrix = data["dist_matrix_Cholet_pb1_bis.pickle"]

    # Ensure solution indices are integers
    solution = solution.astype(int)
    
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

def load_data(folder: str, data :dict) -> dict:
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

def check_validity(lst, data):
    initial_solution = data["init_sol_Cholet_pb1_bis.pickle"]
    return len(set(lst)) == len(initial_solution)

# enter here the solution to test validity and value
solution = [0, 100, 207, 101, 184, 115, 149, 173, 201, 174, 188, 73, 150, 190, 30, 181, 148, 3, 208, 202, 9, 145, 51, 52, 53, 65, 138, 55, 56, 139, 57, 129, 58, 59, 60, 61, 185, 195, 62, 63, 54, 196, 194, 197, 6, 7, 193, 8, 218, 198, 119, 176, 64, 96, 5, 93, 121, 128, 127, 189, 15, 94, 23, 25, 26, 27, 28, 24, 126, 125, 124, 123, 95, 103, 104, 29, 116, 67, 11, 186, 10, 1, 2, 199, 105, 34, 132, 131, 130, 35, 36, 37, 38, 39, 106, 136, 135, 134, 107, 16, 40, 122, 133, 17, 18, 19, 108, 172, 169, 168, 167, 43, 44, 45, 46, 47, 48, 49, 166, 165, 164, 163, 221, 229, 147, 4, 12, 187, 68, 191, 146, 117, 69, 158, 118, 159, 70, 144, 143, 41, 13, 222, 14, 142, 227, 120, 42, 175, 20, 137, 192, 141, 140, 50, 21, 200, 22, 31, 32, 33, 71, 72, 66, 74, 152, 156, 225, 217, 215, 209, 231, 114, 179, 180, 110, 92, 157, 111, 171, 228, 170, 161, 183, 220, 182, 91, 206, 230, 79, 162, 98, 99, 203, 154, 153, 102, 205, 151, 226, 75, 97, 80, 81, 82, 83, 84, 210, 85, 86, 87, 88, 78, 89, 90, 212, 219, 213, 214, 216, 204, 211, 155, 177, 178, 109, 77, 160, 224, 113, 76, 112, 223, 232]
data = {}
load_data("input_data/Probleme_Cholet_1_bis", data)
print("Solution is valid:", check_validity(solution, data) and verify_calculate_weight(np.array(solution), data))
print(calculate_total_dist(np.array(solution), data))