import os
import sys
import pickle
import numpy as np
from itertools import combinations, islice
import multiprocessing
import time


INFINITY = sys.maxsize
data: dict = {}
num_processes = (multiprocessing.cpu_count() // 2) + 2
time_limit = 600

def load_data(folder: str) -> dict:
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  # Convert data to NumPy array


def verify_calculate_weight(solution: np.ndarray, data) -> bool:

    weights = data["weight_Cholet_pb1_bis.pickle"]
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
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  # Calculate total distance using NumPy array indexing
    return total_dist

def total_distance(solution: np.ndarray, data) -> int:
    if verify_calculate_weight(solution, data):
        return calculate_total_dist(solution, data)
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
    return new_solution.copy()

def three_opt_parallel(solution, best_distance, segments, result_queue, start_time, data):
    print(f"Process {os.getpid()} is doing 3-opt")
    best_distance_local = best_distance
    new_best_solution = solution.copy()
    found_better = False
    for i, j, k in segments:
        if time.time() - start_time > time_limit:
            print(f"Process {os.getpid()} time limit exceeded")
            return
        new_solution = three_opt_swap(solution, i, j, k)
        new_distance = total_distance(new_solution, data)
        if new_distance < best_distance_local:
            print(f"Process {os.getpid()} found a better solution, distance delta: {best_distance - new_distance}")
            new_best_solution = new_solution
            best_distance_local = new_distance
            found_better = True
    if found_better:
        result_queue.put((new_best_solution, best_distance_local))
        print(f"Process {os.getpid()} finished")
    

def four_opt_parallel(solution, best_distance, segments, result_queue, better_solution_found, lock, start_time, data):
    print(f"Process {os.getpid()} is doing 4-opt")
    for i, j, k in segments:
        for l in range(k + 1, len(solution) - 1):
            if time.time() - start_time > time_limit:
                print(f"Process {os.getpid()} time limit exceeded")
                return
            with lock:
                if better_solution_found.value:
                    return
            new_solution = four_opt_swap(solution, i, j, k, l)
            new_distance = total_distance(new_solution, data)
            if new_distance < best_distance:
                print(f"Process {os.getpid()} found a better solution, distance delta: {best_distance - new_distance}")
                result_queue.put((new_solution, new_distance))
                with lock:
                    better_solution_found.value = True
                return


def four_opt_swap(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k], solution[i:j], solution[l:]))
    return new_solution.copy()

def three_opt(solution):
    improved = True
    best_distance = total_distance(solution, data)


    segments = get_all_segments(solution)
    np.random.shuffle(segments)
    start = time.time()
    processes = []

    while improved and time.time() - start < time_limit:
        improved = False
        chunk_size = len(segments) // num_processes
        chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]

        if len(segments) % num_processes != 0:
            chunks[-2] += chunks[-1]
            chunks.pop(-1)

        result_queue = multiprocessing.Queue()
        for chunk in chunks:
            process = multiprocessing.Process(target=three_opt_parallel, args=(solution, best_distance, chunk, result_queue, start, data))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        if result_queue.empty() and time.time() - start < time_limit:
            print("!!! No better solution found using 3-opt, trying 4-opt !!!")
            better_solution_found = multiprocessing.Value('b', False)
            lock = multiprocessing.Lock()
            for chunk in chunks:
                process = multiprocessing.Process(target=four_opt_parallel, args=(solution, best_distance, chunk, result_queue, better_solution_found, lock, start, data))
                process.start()
                processes.append(process)
            
            for process in processes:
                process.join()

        while not result_queue.empty():
            new_solution, new_distance = result_queue.get()
            if new_distance < best_distance:
                solution = new_solution
                best_distance = new_distance
                improved = True
        print("Improved distance: ", best_distance)
        
    for process in processes:
        process.terminate()

    if time.time() - start < time_limit:
        print("Time left: ", time_limit - (time.time() - start))
    else:
        print("Time limit exceeded")

    return solution, best_distance


if __name__ == "__main__":


    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]

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
    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(solution, data))
    solution, best_distance = three_opt(solution)
    print("Final solution: ", solution)
    print("Final distance: ", best_distance)

"""
local minimum found with 3-opt 

in around 7 minutes with 6 processes :

    [0,  77, 100, 101, 184, 115,  79,  98,  99,  91, 206, 230, 162, 177, 178, 109,  80,  81, 82,  83,  84, 210,  85,  86,  87,  88,  78, 214,  89, 213,  90, 212, 219, 216, 204, 211, 155,  93,  15, 189, 121, 128, 127,  94,  23,  26,  27,  28,  24,  25, 126, 125, 124, 123, 95, 221, 229,  29,   1,   2, 199, 116,  67, 186,  10,  11,  12, 187 , 68, 191, 146, 117, 69, 163, 103, 104, 105,  34, 132, 131, 130 , 35,  36,  37,  38,  39, 106, 136, 135, 134, 107,  16,  40, 122, 133,  17,  18,  19, 108, 169, 172, 168, 167 , 43 , 44 , 45 , 46 , 47, 48,  49, 166, 165, 164, 147,   4, 158, 118, 144, 143,  41,  13, 222,  14, 142, 227, 120, 42, 175,  58,  59,  60, 192 , 61, 185, 141, 140,  50,   9, 145 , 51,  52,  53,  65,  55, 56, 139,  57, 138, 129, 137, 195,  62,  63,  54, 194, 196, 197,   6,   7, 193,   8, 218, 198, 119, 176,  64,  96,   5, 159,  70,  20,  21, 200,  22,  31, 202 , 32,  72,  66, 207, 160, 224, 113,  76, 156, 225, 209, 217, 215, 231, 114, 149,   3, 208,  33,  71, 173, 201, 174, 188,  73, 150, 190,  30, 181, 148, 220, 110,  92, 157, 111, 180, 179, 171, 228, 170, 161, 183, 182, 203, 154, 153, 102,  74, 205,  75, 152, 151, 226,  97, 112, 223, 232]

distance: 42439

in around 1.5 minutes with 16 processes :

    [0,77,100,101,184,115,79,162,177,178,109,80,81,82,83,84,210,85
    ,86,87,88,78,214,89,213,90,212,219,216,204,211,155,93,121,128,127
    ,189,15,94,23,26,27,28,24,25,126,125,124,123,95,221,229,29,1
    ,2,199,116,67,186,10,11,12,187,68,191,146,117,69,163,103,104,105
    ,34,132,131,130,35,36,37,38,39,106,136,135,134,107,16,40,122,133
    ,17,18,19,108,169,172,168,167,43,44,45,46,47,48,49,166,165,164
    ,147,4,158,118,58,59,61,185,141,140,50,9,145,,51,52,53,65,55
    ,56,139,57,138,129,137,60,192,120,42,175,96,5,159,144,143,41,13
    ,222,14,142,227,195,62,63,54,194,196,197,6,7,193,8,218,198,119
    ,176,64,70,20,21,200,22,31,202,32,33,71,72,66,74,205,75,97
    ,112,152,151,226,156,225,209,217,215,231,114,149,173,201,174,188,73,150
    ,190,30,181,148,3,208,220,110,92,157,111,180,179,171,228,170,161,183
    ,182,91,206,230,98,99,203,154,153,102,207,160,224,113,76,223,232]

distance: 41497



With 6 processes (and better chunking and process returning only if delta  > 100):
Time left:  321
Final solution:  [  0  80  81  82  83  84 210  85  86  87  88  78 214  89 213  90 212 219
     216 204 211 155 177  93 121 128 127 189  15  94  23  26  27  28  24  25
     126 125 124 123  95 221 229  29 116 186  10   1   2 199  67  11  12 187
      68 191 146 117  69 163 103 104 105  34 132 131 130  35  36  37  38  39
     106 136 135 134 107  16  40 122 133  17  18  19 108 169 172 168 167  43
      44  45  46  47  48  49 166 165 164 147   4 158 118 144 143  41  13 222
      14 142 227 195  62  63  54 194 196 197   6   7 193   8 218 198 119 176
      64  96   5 159  70  20   9 145  51  52  53  65  55  56 139  57 138 129
     137 120  42 175  58  59  60 192  61 185 141 140  50  21 200  22  31 202
      32  33  71  72  66  74 205  75 152 151 226  97 112 156 225 209 217 215
     231 114 149 173 201 174 188  73 150 190  30 181 148   3 208 220 110  92
     157 111 180 179 171 228 170 161 183 182 178 109  77 100 207 101 184 115
      79 162  98  99  91 206 230 203 154 153 102 160 224 113  76 223 232]
Final distance:  42081


With 6 processes, better chuncking and finding the best solution for every chunk no 4-opt:
Time left:  339
Final solution:  [  0  80  81  82  83  84 210  85  86  87  88  78 214  89 213  90 212 219
 216 204 211 155 177 169 108 172 168  16  17  18  19  40 122 133 167  34
 132 131 130  35  36  37  38  39  29 116 186  10   1   2 199  67  11  12
 187  68 191 146 117  69 163 103 104 105 106 136 135 134 107  43  44  45
  46  47  48  49 166 165 164 147   4 158  26  27  28  24  25 126 125 124
 123  95 221 229  93 121 128 127 189  15  94  23 118 159  70  58  59 195
  62  63  54 194 196 197   6   7 193   8 218 198 119 176  64  96   5 144
 143  41  13 222  14 142 227 120  42 175  20   9 145  51  52  53  65  55
  56 139  57 138 129 137  60 192  61 185 141 140  50  21 200  22  31 202
  32  33  71  72  66  74 205  97 112 152 151 226  75 225 209 217 215 231
 114 149 173 201 174 188  73 150 190  30 181 148   3 208 220 110  92 157
 111 180 179 171 228 170 161 183 182 178 109  77 100 207 101 184 115  79
 162  98  99  91 206 230 203 154 153 102 160 224 113  76 156 223 232]
Final distance:  42427


With 6 processes, better chuncking and finding the best solution for every chunk and delta > 10 for 3-opt:
Time limit exceeded
Final solution:  [  0  77 100 207 101 184 115  79 177 178 109  80  81  82  83  84 210  85
  86  87  88  78 214  89 213  90 212 219 216 204 211 155 169 108 172 168
  43  44  45  46  47  48  49 166 165  29 116 186  10   1   2 199  67  11
 158 189  15 127  94  23  26  27  28  24  25 126 125 124 123  95 103 104
 105 106 136 135 134 107  16  17  18  19  40 122 133 167  34 132 131 130
  35  36  37  38  39 164 147   4  12 187  68 191 146 117  69 163 221 229
  93 121 128 118 159  70  20 137 120 222  14 142 227 195  62  63  54 194
 196 197   6   7 193   8 218 198 119 176  64  96   5 144 143  41  13  42
 175  58  59  60 192  61 185 141 140  50   9 145  51  52  53  65  55  56
 139  57 138 129  21 200  22  31 202  32  33  71  72  66  74 205  97 112
 152 151 226  75 225 209 217 215 231 114 149 173 201 174 188  73 150 190
  30 181 148   3 208 220 110  92 157 111 180 179 171 228 170 161 183 182
  91 206 230 162  98  99 203 154 153 102 160 224 113  76 156 223 232]
Final distance:  41866

With 6 processes, better chuncking and finding the best solution for every chunk with 4-opt:
Final solution:  [  0  77 100 207 101 184 115  79 177 178 109  80  81  82  83  84 210  85
  86  87  88  78 214  89 213  90 212 219 216 204 211 155 169 108 172 168
  16  17  18  19  40 122 133 167  34 132 131 130  35  36  37  38  39  29
 116 186  10   1   2 199  67  11  12 187  68 191 146 117  69 104 105 106
 136 135 134 107  43  44  45  46  47  48  49 166 165 164 163 103 147   4
 158  26  27  28  24  25 126 125 124 123  95 221 229  93 121 128 127 189
  15  94  23 118 144 143  41  13 222  14 142 227 195  62  63  54 194 196
 197   6   7 193   8 218 198 119 176  64  96   5 159  70  20   9 145  51
  52  53  65  55  56 139  57 138 129 137 120  42 175  58  59  60 192  61
 185 141 140  50  21 200  22  31 202  32  33  71  72  66  74 205  97 112
 152 151 226  75 225 209 217 215 231 114 149 173 201 174 188  73 150 190
  30 181 148   3 208 220 110  92 157 111 180 179 171 228 170 161 183 182
  91 206 230 162  98  99 203 154 153 102 160 224 113  76 156 223 232]
Final distance:  41856
"""
