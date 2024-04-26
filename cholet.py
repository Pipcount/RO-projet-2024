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
     



if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("usage: python3 main.py <folder>")
