import pickle, sys, os


def main(folder):
    data = load_data(folder)
    calculate_total_dist(data)


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
    print(init_sol)
    print()
    print(dist_matrix)
    print()
    for i in range(len(init_sol)):
        node = init_sol[i]
        total_dist += dist_matrix[node][node]
        if i < len(init_sol) - 1:
            next_node = init_sol[i + 1]
            total_dist += dist_matrix[node][next_node]
    print(total_dist)
    return total_dist


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("usage: python3 main.py <folder>")
