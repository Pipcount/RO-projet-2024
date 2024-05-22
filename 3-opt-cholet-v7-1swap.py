import os
import sys
import pickle
import numpy as np
from itertools import combinations
import multiprocessing
import time


INFINITY = sys.maxsize
data: dict = {}
num_processes = 16 #(multiprocessing.cpu_count() // 2)
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

def total_distance(solution: np.ndarray, data, best_distance) -> int:
    distance = calculate_total_dist(solution, data)
    if distance < best_distance:
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

def three_opt_parallel(segments, best_solution, start_time, shape, solution_lock, event, queue):
    print(f"Process {os.getpid()} started")

    while time.time() - start_time < time_limit:
        print(f"Process {os.getpid()} started iteration")
        best_new_distance = INFINITY
        with solution_lock:
            best_solution_for_now = np.frombuffer(best_solution, dtype='d').reshape(shape)
        best_new_solution = np.array([])
        for i, j, k in segments:
            if time.time() - start_time > time_limit:
                return
            new_solution = three_opt_swap(best_solution_for_now, i, j, k)
            new_distance = total_distance(new_solution, data, best_new_distance)
            if new_distance < best_new_distance:
                best_new_distance = new_distance
                best_new_solution = new_solution
        queue.put((best_new_distance, best_new_solution))

        print(f"Process {os.getpid()} finished iteration")
        event.wait()


def update_solution(best_solution, best_distance, start_time, solution_lock, event, queue):
    print("Solution updater started")
    while time.time() - start_time < time_limit:
        event.clear()
        if queue.qsize() == num_processes:
            print("All processes finished iteration, updating best solution")
            best_queue_solution = (INFINITY, None)
            for _ in range(num_processes):
                distance, solution = queue.get()
                if distance < best_queue_solution[0]:
                    best_queue_solution = (distance, solution)
            with best_distance.get_lock() and solution_lock:
                print(f"Current best distance: {best_queue_solution[0]}, delta: {best_distance.value - best_queue_solution[0]}, time: {time.time() - start_time}")
                best_distance.value = best_queue_solution[0]
                best_solution[:] = best_queue_solution[1].flatten()
            event.set()
        time.sleep(1)

    event.set()


def three_opt(solution):
    start = time.time()
    best_distance = total_distance(solution, data, INFINITY)
    segments = get_all_segments(solution)
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
    queue = multiprocessing.Queue(maxsize=num_processes)

    for chunk in chunks:
        p = multiprocessing.Process(target=three_opt_parallel, args=(chunk, best_solution, start, solution_shape, best_solution_lock, event, queue))
        p.start()
        processes.append(p)

    solution_updater = multiprocessing.Process(target=update_solution, args=(best_solution, best_distance_value, start, best_solution_lock, event, queue))
    solution_updater.start()
    processes.append(solution_updater)

    for p in processes:
        p.join()

    with best_solution_lock:
        solution_found = np.frombuffer(best_solution, dtype='d').reshape(solution_shape)
    with best_distance_value.get_lock():
        best_distance_found = best_distance_value.value

    return solution_found, best_distance_found



if __name__ == "__main__":

    print("Number of processes: ", num_processes)
    load_data("input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]

    print("Initial solution: ", solution)
    print("Initial distance: ", total_distance(solution, data, INFINITY))
    best_solution, best_distance = three_opt(solution)
    print("Best solution: ", list(best_solution))
    print("Best distance: ", best_distance)


"""
    Number of processes:  16
    CPU Frequency: ~5GHz
    Time limit: 60s
    Best solution:  [0.0, 80.0, 81.0, 82.0, 83.0, 84.0, 210.0, 85.0, 86.0, 87.0, 88.0, 78.0, 89.0, 213.0, 90.0, 212.0, 219.0, 214.0, 216.0, 204.0, 211.0, 155.0, 177.0, 178.0, 109.0, 169.0, 168.0, 16.0, 40.0, 122.0, 133.0, 17.0, 18.0, 19.0, 108.0, 172.0, 168.0, 167.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 107.0, 16.0, 17.0, 18.0, 19.0, 40.0, 122.0, 133.0, 167.0, 166.0, 34.0, 132.0, 131.0, 130.0, 35.0, 36.0, 37.0, 38.0, 39.0, 165.0, 29.0, 116.0, 186.0, 10.0, 1.0, 2.0, 67.0, 11.0, 117.0, 221.0, 229.0, 163.0, 26.0, 27.0, 28.0, 24.0, 25.0, 126.0, 125.0, 124.0, 123.0, 95.0, 103.0, 104.0, 105.0, 34.0, 132.0, 131.0, 130.0, 37.0, 38.0, 39.0, 106.0, 136.0, 135.0, 134.0, 166.0, 165.0, 29.0, 199.0, 164.0, 147.0, 4.0, 12.0, 68.0, 191.0, 146.0, 187.0, 69.0, 158.0, 15.0, 127.0, 94.0, 23.0, 118.0, 159.0, 70.0, 20.0, 21.0, 22.0, 31.0, 32.0, 33.0, 71.0, 72.0, 66.0, 207.0, 112.0, 231.0, 114.0, 149.0, 173.0, 201.0, 174.0, 188.0, 73.0, 150.0, 190.0, 30.0, 181.0, 148.0, 202.0, 9.0, 145.0, 51.0, 52.0, 53.0, 65.0, 55.0, 56.0, 139.0, 57.0, 138.0, 129.0, 58.0, 59.0, 195.0, 62.0, 63.0, 54.0, 194.0, 196.0, 197.0, 6.0, 7.0, 193.0, 8.0, 218.0, 198.0, 119.0, 176.0, 64.0, 96.0, 5.0, 93.0, 121.0, 189.0, 128.0, 144.0, 143.0, 41.0, 13.0, 222.0, 14.0, 142.0, 227.0, 120.0, 42.0, 175.0, 137.0, 60.0, 192.0, 61.0, 185.0, 141.0, 140.0, 50.0, 200.0, 3.0, 208.0, 220.0, 110.0, 92.0, 157.0, 111.0, 180.0, 179.0, 171.0, 228.0, 170.0, 161.0, 183.0, 182.0, 203.0, 154.0, 160.0, 224.0, 113.0, 76.0, 223.0, 232.0]
    Best distance:  38844.0
"""

"""
    Number of processes:  16
    CPU Frequency: ~5GHz
    Time limit: 600s
    Best solution:  [0.0, 80.0, 81.0, 82.0, 83.0, 84.0, 210.0, 85.0, 86.0, 87.0, 88.0, 78.0, 89.0, 90.0, 212.0, 219.0, 213.0, 214.0, 216.0, 204.0, 211.0, 155.0, 177.0, 178.0, 109.0, 169.0, 168.0, 16.0, 40.0, 122.0, 122.0, 133.0, 17.0, 108.0, 172.0, 172.0, 168.0, 167.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 166.0, 165.0, 29.0, 1.0, 2.0, 199.0, 116.0, 67.0, 186.0, 10.0, 11.0, 117.0, 221.0, 229.0, 147.0, 4.0, 12.0, 68.0, 191.0, 146.0, 187.0, 69.0, 163.0, 26.0, 27.0, 28.0, 24.0, 25.0, 126.0, 125.0, 124.0, 123.0, 93.0, 121.0, 189.0, 128.0, 127.0, 15.0, 94.0, 95.0, 103.0, 104.0, 105.0, 34.0, 132.0, 131.0, 130.0, 37.0, 38.0, 39.0, 106.0, 107.0, 16.0, 17.0, 18.0, 19.0, 40.0, 19.0, 18.0, 133.0, 167.0, 136.0, 135.0, 134.0, 166.0, 34.0, 132.0, 131.0, 130.0, 35.0, 36.0, 37.0, 38.0, 39.0, 165.0, 164.0, 158.0, 23.0, 118.0, 144.0, 143.0, 41.0, 13.0, 222.0, 14.0, 141.0, 140.0, 50.0, 22.0, 31.0, 32.0, 33.0, 71.0, 72.0, 66.0, 207.0, 112.0, 231.0, 114.0, 149.0, 173.0, 201.0, 174.0, 188.0, 73.0, 150.0, 190.0, 30.0, 181.0, 148.0, 202.0, 137.0, 60.0, 120.0, 42.0, 175.0, 58.0, 59.0, 192.0, 61.0, 185.0, 142.0, 227.0, 195.0, 62.0, 63.0, 54.0, 194.0, 196.0, 197.0, 6.0, 7.0, 8.0, 218.0, 198.0, 119.0, 193.0, 176.0, 64.0, 96.0, 5.0, 159.0, 70.0, 20.0, 9.0, 145.0, 51.0, 52.0, 53.0, 65.0, 55.0, 56.0, 139.0, 57.0, 138.0, 129.0, 21.0, 200.0, 3.0, 208.0, 220.0, 110.0, 92.0, 157.0, 111.0, 180.0, 179.0, 171.0, 228.0, 170.0, 161.0, 183.0, 182.0, 203.0, 154.0, 160.0, 224.0, 113.0, 76.0, 223.0, 232.0]
    Best distance:  38576.0
"""