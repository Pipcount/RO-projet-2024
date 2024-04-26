import pickle, sys, os


def main(folder):
    data = load_data(folder)
    calculate_total_dist(data)
    calculate_weight(data)
    calculate_total_time(data)

def load_data(folder : str) -> dict: 
    data = {}
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = pickle.load(f)
    return data

def calculate_total_dist(data: dict) -> int:
    total_dist = 0
    init_sol = data["init_sol_Cholet_pb1_bis.pickle"]     
    dist_matrix = data["dist_matrix_Cholet_pb1_bis.pickle"]
    for i in range(len(init_sol)):
        node = init_sol[i]
        total_dist += dist_matrix[node][node]
        if i < len(init_sol) - 1:
            next_node = init_sol[i + 1]
            total_dist += dist_matrix[node][next_node]
    print(total_dist)
    return total_dist

def calculate_weight(data : dict):
    weight = 0
    weights = data["weight_Cholet_pb1_bis.pickle"]
    init_sol = data["init_sol_Cholet_pb1_bis.pickle"]     
    for node in init_sol:
        weight += weights[node]
        if weight < 0:
            weight = 0
        print(weight)
        if weight > 5850:
            print("aaaaaaaaaaaaa")

def calculate_total_time(data : dict) -> int:
    total_time = 0
    init_sol = data["init_sol_Cholet_pb1_bis.pickle"]     
    time_matrix = data["dur_matrix_Cholet_pb1_bis.pickle"]
    time_collect = data["temps_collecte_Cholet_pb1_bis.pickle"]
    for i in range(len(init_sol)):
        node = init_sol[i]
        total_time += time_matrix[node][node]
        total_time += time_collect[node]
        if i < len(init_sol) - 1:
            next_node = init_sol[i + 1]
            total_time += time_matrix[node][next_node]
    time_in_hours = total_time // 3600
    print(time_in_hours)
    return total_time

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("usage: python3 main.py <folder>")
