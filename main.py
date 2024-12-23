import pickle
import sys, os

def load_data(folder : str) -> dict: 
    data = {}
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = pickle.load(f)
    return data


def print_data(data : dict) -> None:
    for key in data:
        print(key)
        print(data[key])
        print("\n")

def main(folder):
    data = load_data(folder)
    print_data(data)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("usage: python3 main.py <folder>")
