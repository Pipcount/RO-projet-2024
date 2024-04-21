import pickle
import sys, os

def load_data(folder):
    data = {}
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = pickle.load(f)
    return data

def main(folder):
    data = load_data(folder)
    print(data)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("usage: python3 main.py <folder>")
