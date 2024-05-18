import os
import sys
import pickle
import numpy as np
from itertools import combinations, islice
import multiprocessing
import time


INFINITY = sys.maxsize
data: dict = {}
num_processes = (multiprocessing.cpu_count() // 2)
time_limit = 600

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


def three_opt_swap1(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j], solution[k:]))
    return new_solution.copy()

def three_opt_swap2(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k][::-1], solution[i:j], solution[k:]))
    return new_solution.copy()

def three_opt_swap3(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j][::-1], solution[k:]))
    return new_solution.copy()

def three_opt_swap4(solution, i, j, k):
    new_solution = np.concatenate((solution[:i], solution[j:k][::-1], solution[i:j][::-1], solution[k:]))
    return new_solution.copy()


def three_opt_parallel(segments, best_solution, best_distance, start_time, lock, shape, event):
    print(f"Process {os.getpid()} started")
    stop_iteration = len(segments)
    current_best_distance = INFINITY
    waiting = False
    four_opt = False
    with lock:
        current_solution = np.frombuffer(best_solution, dtype='d').reshape(shape)
    idx = 0
    while time.time() - start_time < time_limit:

        with lock:
            if best_distance.value < current_best_distance:
                current_solution = np.frombuffer(best_solution, dtype='d').reshape(shape)
                current_best_distance = best_distance.value
                stop_iteration = idx
                if four_opt:
                    print(f"Process {os.getpid()} is now doing 3-opt")
                four_opt = False
            elif stop_iteration == idx:
                if four_opt:
                    print(f"Process {os.getpid()} reached the end of the segment list and is waiting for the other processes to find a better solution")
                    waiting = True
                    four_opt = False
                else:
                    print(f"Process {os.getpid()} is now doing 4-opt")
                    four_opt = True
        if waiting:
            while not event.is_set():
                if time.time() - start_time > time_limit:
                    print(f"Process {os.getpid()} time limit exceeded")
                    return
                time.sleep(1)
            waiting = False
            time.sleep(1)
            event.clear()
            print(f"Process {os.getpid()} woke up")

        i, j, k = segments[idx]
        if not four_opt:
            for swap in [three_opt_swap1, three_opt_swap2, three_opt_swap3, three_opt_swap4]:
                new_solution = swap(current_solution.copy(), i, j, k)
                new_distance = total_distance(new_solution, data)
                if new_distance < current_best_distance:
                    print(f"Process {os.getpid()} found a better solution, new distance: {new_distance}, that is {current_best_distance - new_distance} better")
                    with lock:
                        if new_distance < best_distance.value:
                            np.copyto(current_solution, new_solution)
                            best_distance.value = new_distance
                            best_solution[:] = new_solution.flatten()
                            event.set()
                    stop_iteration = idx
                    break
        else:
            for l in range(k, len(current_solution) - 1):
                for swap in [four_opt_swap1, four_opt_swap2, four_opt_swap3, four_opt_swap4, four_opt_swap5, four_opt_swap6, four_opt_swap7, four_opt_swap8, four_opt_swap9]:
                    new_solution = swap(current_solution.copy(), i, j, k, l)
                    new_distance = total_distance(new_solution, data)
                    if new_distance < current_best_distance:
                        print(f"Process {os.getpid()} found a better solution, new distance: {new_distance}, that is {current_best_distance - new_distance} better") 
                        with lock:
                            if new_distance < best_distance.value:
                                np.copyto(current_solution, new_solution)
                                best_distance.value = new_distance
                                best_solution[:] = new_solution.flatten()
                                event.set()
                        stop_iteration = idx
                        break
        if idx == len(segments) - 1:
            idx = 0
        else:
            idx += 1
    print(f"Process {os.getpid()} finished")


def four_opt_parallel(segments, best_solution, best_distance, start_time, lock, shape, event):
    print(f"Process {os.getpid()} started")
    stop_iteration = len(segments)
    current_best_distance = INFINITY
    waiting = False
    with lock:
        current_solution = np.frombuffer(best_solution, dtype='d').reshape(shape)
    idx = 0
    while time.time() - start_time < time_limit:

        with lock:
            if best_distance.value < current_best_distance:
                current_solution = np.frombuffer(best_solution, dtype='d').reshape(shape)
                current_best_distance = best_distance.value
                stop_iteration = idx
            elif stop_iteration == idx:
                print(f"Process {os.getpid()} reached the end of the segment list and is waiting for the other processes to find a better solution")
                waiting = True
        if waiting:
            while not event.is_set():
                if time.time() - start_time > time_limit:
                    print(f"Process {os.getpid()} time limit exceeded")
                    return
                time.sleep(1)
            waiting = False
            time.sleep(1)
            event.clear()
            print(f"Process {os.getpid()} woke up")

        i, j, k = segments[idx]
        for l in range(k, len(current_solution) - 1):
            for swap in [four_opt_swap1, four_opt_swap2, four_opt_swap3, four_opt_swap4, four_opt_swap5, four_opt_swap6, four_opt_swap7, four_opt_swap8, four_opt_swap9]:
                new_solution = swap(current_solution.copy(), i, j, k, l)
                new_distance = total_distance(new_solution, data)
                if new_distance < current_best_distance:
                    print(f"Process {os.getpid()} found a better solution, new distance: {new_distance}, that is {current_best_distance - new_distance} better")
                    with lock:
                        if new_distance < best_distance.value:
                            np.copyto(current_solution, new_solution)
                            best_distance.value = new_distance
                            best_solution[:] = new_solution.flatten()
                            event.set()
                    stop_iteration = idx
                    break
        if idx == len(segments) - 1:
            idx = 0
        else:
            idx += 1
    print(f"Process {os.getpid()} finished")

def four_opt_swap1(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap2(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap3(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k][::-1], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap4(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def four_opt_swap5(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k][::-1], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap6(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l], solution[j:k][::-1], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def four_opt_swap7(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k][::-1], solution[i:j], solution[l:]))
    return new_solution.copy()

def four_opt_swap8(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def four_opt_swap9(solution, i, j, k, l):
    new_solution = np.concatenate((solution[:i], solution[k:l][::-1], solution[j:k][::-1], solution[i:j][::-1], solution[l:]))
    return new_solution.copy()

def three_opt(solution, seed):
    start = time.time()
    best_distance = total_distance(solution, data)
    segments = get_all_segments(solution)
    np.random.seed(seed)
    np.random.shuffle(segments)
    print(f"Seed: {np.random.get_state()[1][0]}")
    processes = []

    chunk_size = len(segments) // num_processes
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    if len(segments) % num_processes != 0:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)

    solution_shape = solution.shape
    best_solution = multiprocessing.Array('d', solution.flatten(), lock=False)
    best_distance_value = multiprocessing.Value('d', best_distance)
    lock = multiprocessing.Lock()
    event = multiprocessing.Event()

    for chunk in chunks:
        process = multiprocessing.Process(target=three_opt_parallel, args=(chunk, best_solution, best_distance_value, start, lock, solution_shape, event))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    for process in processes:
        process.terminate()

    best_solution_np = np.frombuffer(best_solution, dtype='d').reshape(solution_shape)
    return best_solution_np, best_distance_value.value


if __name__ == "__main__":


    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]

    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(solution, data))
    best_seed = 0
    best_distance = INFINITY
    best_solution = []
    while True:
        seed = np.random.randint(0, 10000)
        new_solution, new_distance = three_opt(solution, seed)
        if new_distance < best_distance:
            best_distance = new_distance
            best_solution = new_solution
            best_seed = seed
            print(f"New best distance: {best_distance}, seed: {best_seed}, solution: {best_solution}")
            with open("output.txt", 'a') as f:
                f.write(f"New best distance: {best_distance}, seed: {best_seed}, solution: {best_solution}\n")

"""
12 processes


New best distance: 41566.0, seed: 2810, solution: [  0.  77. 100. 101. 184. 115.  79.  98.  99.  91. 206. 230. 162. 177.
 178. 109.  80.  81.  82.  83.  84. 210.  85.  86.  87.  88.  78. 214.
  89. 213.  90. 212. 219. 216. 204. 211. 155.  93. 121. 128. 127. 189.
  15.  94.  23.  26.  27.  28.  24.  25. 126. 125. 124. 123.  95. 103.
 147.   4.  12. 187.  68. 191. 146. 117.  69. 163. 221. 229. 104. 105.
  34. 132. 131. 130.  35.  36.  37.  38.  39. 106. 136. 135. 134. 107.
  16.  40. 122. 133.  17.  18.  19. 108. 169. 172. 168. 167.  43.  44.
  45.  46.  47.  48.  49. 166. 165.  29. 116.  67.  11. 186.  10.   1.
   2. 199. 164. 158. 118. 159. 144. 143.  41.  13. 222.  14. 142. 227.
 195.  62.  63.  54. 194. 196. 197.   6.   7. 193.   8. 218. 198. 119.
 176.  64.  96.   5.  70.  20.   9. 145.  51.  52.  53.  65.  55.  56.
 139.  57. 138. 129. 137.  60. 192.  61. 120.  42. 175.  58.  59. 185.
 141. 140.  50.  21. 200.  22.  31. 202.  32.  33.  71.  72.  66.  74.
 205. 225. 209. 217. 215. 231. 114. 149. 173. 201. 174. 188.  73. 150.
 190.  30. 181. 148.   3. 208. 220. 110.  92. 157. 111. 180. 179. 171.
 228. 170. 161. 183. 182. 203. 154. 153. 102. 207. 160. 224. 113.  76.
 156.  97. 112. 152. 151. 226.  75. 223. 232.]

New best distance: 40916.0, seed: 7413, solution: [  0.  77. 100. 207. 101. 184. 115.  79.  98.  99.  91. 206. 230. 162.
 177. 178. 109.  80.  81.  82.  83.  84. 210.  85.  86.  87.  88.  78.
 214.  89. 213.  90. 212. 219. 216. 204. 211. 155. 169. 108. 172. 168.
  16.  17.  18.  19.  40. 122. 133. 167.  34. 132. 131. 130.  35.  36.
  37.  38.  39.  29.   1.   2. 199. 116.  67. 186.  10.  11. 117.  69.
 104. 105. 106. 136. 135. 134. 107.  43.  44.  45.  46.  47.  48.  49.
 166. 165. 164. 147.   4.  12.  68. 191. 146. 187. 221. 229. 163.  26.
  27.  28.  24.  25. 126. 125. 124. 123.  95. 103. 158. 144. 143.  41.
  13. 222.  14. 142. 227. 195.  62.  63.  54. 194. 196. 197.   6.   7.
 193.   8. 218. 198. 119. 176.  64.  96.   5.  93. 121. 128. 127. 189.
  15.  94.  23. 118. 159.  70.  58.  59. 120.  42. 175.  20.   9. 145.
  51.  52.  53.  65.  55.  56. 139.  57. 138. 129. 137.  60. 192.  61.
 185. 141. 140.  50.  21. 200.  22.  31. 202.  32.  33.  71.  72.  66.
 114. 231. 209. 217. 215. 156. 225.  97. 160. 224. 113.  76. 112. 152.
 151. 226. 149. 173. 201. 174. 188.  73. 150. 190.  30. 181. 148.   3.
 208. 220. 110.  92. 157. 111. 180. 179. 171. 228. 170. 161. 183. 182.
 203. 154. 153. 102.  74. 205.  75. 223. 232.]

"""
