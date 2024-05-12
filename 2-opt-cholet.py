import os, sys, pickle

INFINITY = sys.maxsize
data : dict = {}

def load_data(folder : str) -> dict: 
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = pickle.load(f)

def verify_calculate_weight(solution: list) -> bool:
    weight = 0
    weights = data["weight_Cholet_pb1_bis.pickle"]  
    for node in solution:
        weight += weights[node]
        if weight < 0:
            weight = 0
        if weight > 5850:
            return False
    return True

def calculate_total_dist(solution : list) -> int:
    total_dist = 0
    dist_matrix = data["dist_matrix_Cholet_pb1_bis.pickle"]
    for i in range(len(solution)):
        node = solution[i]
        # total_dist += dist_matrix[node][node] : ajouter la diagonale de la matrice de distance
        if i < len(solution) - 1:
            next_node = solution[i + 1]
            total_dist += dist_matrix[node][next_node]
    # print(total_dist)
    return total_dist

def total_distance(solution : list) -> int:
    if verify_calculate_weight(solution):
        return calculate_total_dist(solution)
    return INFINITY

def two_opt(solution):
    improved = True
    best_distance = total_distance(solution)
    
    while improved:
        improved = False
        for i in range(1, len(solution) - 2):
            for j in range(i + 1, len(solution)):
                if j - i == 1:
                    continue  # No reverse possible for adjacent edges
                new_solution = solution[:]
                new_solution[i:j] = reversed(new_solution[i:j])
                new_distance = total_distance(new_solution)
                if new_distance < best_distance:
                    solution = new_solution
                    best_distance = new_distance
                    improved = True
        if improved:
            print("Improved distance: ", best_distance)
    
    return solution, best_distance

def three_opt(solution):
    improved = True
    best_distance = total_distance(solution)
    
    while improved:
        improved = False
        for i in range(1, len(solution) - 4):
            for j in range(i + 2, len(solution) - 2):
                for k in range(j + 2, len(solution)):
                    new_solution = solution[:]
                    new_solution[i:j], new_solution[j:k] = solution[j:k][::-1], solution[i:j][::-1]
                    new_distance = total_distance(new_solution)
                    if new_distance < best_distance:
                        solution = new_solution
                        best_distance = new_distance
                        improved = True
        if improved:
            print("Improved distance: ", best_distance)
    
    return solution, best_distance
if __name__ == "__main__":
    load_data("input_data/Probleme_Cholet_1_bis")
    initial_solution = data["init_sol_Cholet_pb1_bis.pickle"]
    # initial_solution = [i for i in range(0, 233)]
    print("Initial solution: ", initial_solution)
    print("Initial distance: ", total_distance(initial_solution))
    # improved_solution, improved_distance = two_opt(initial_solution)
    improved_solution, improved_distance = three_opt(initial_solution)
    print("Improved solution: ", improved_solution)
    print("Improved distance: ", improved_distance)
