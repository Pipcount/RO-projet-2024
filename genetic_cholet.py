import pickle, sys, os
import random

INFINITY = 50000
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

def fitness_function(solution : list) -> int:
    if verify_calculate_weight(solution):
        return calculate_total_dist(solution)
    return calculate_total_dist(solution) * INFINITY

def ordered_crossover(parent1 : list, parent2 : list) -> list:
    size = len(parent1) - 2  # Exclude first and last elements
    # Step 1: Select a random subset from parent1 (excluding first and last elements)
    start, end = sorted(random.sample(range(1, size+1), 2))  # Start from 1 and end at size+1 to exclude first and last elements
    subset = parent1[start:end+1]
    # Step 2: Copy this subset directly to the child, in the same positions
    child = [None]*(size+2)  # Include space for first and last elements
    child[0] = parent1[0]  # Copy first element
    child[-1] = parent1[-1]  # Copy last element
    child[start:end+1] = subset
    # Step 3: Starting from the end of the subset in parent2, copy the remaining elements to the child
    # in the order they appear in parent2, wrapping around to the start of the list if necessary
    pointer = end + 1
    if pointer >= size+2:  # Wrap around to the start of the list
        pointer = 1  # Start from 1 to exclude first element
    for element in parent2:
        if element not in subset and element != parent1[0] and element != parent1[-1]:  # Exclude first and last elements
            while child[pointer] is not None:
                pointer += 1
                if pointer >= size+1:  # Wrap around to the start of the list, excluding the last element
                    pointer = 1  # Start from 1 to exclude first element
            child[pointer] = element
    return child

def tsp_permute(solution: list) -> list:
    solution = solution.copy()
    new_solution = solution.copy()
    a = random.randint(0, len(solution)-2)
    b = random.randint(a+2, len(solution))
    solution[a:b] = reversed(new_solution[a:b])
    return solution

def generate_initial_population(initial_solution: list, population_size: int) -> list:
    population = []
    population.append(initial_solution.copy())
    for _ in range(population_size - 1):
        population.append(tsp_permute(initial_solution))
    return population

def genetic_algorithm(initial_solution : list, population_size : int, max_generations : int, tournament_size : int, children_size : int, keepParents : bool) -> list:
    population = generate_initial_population(initial_solution, population_size)

    for i in range(max_generations):
        print("Starting generation " + str(i+1))
        population = find_best_solutions(population, tournament_size)

        new_generation = []

        # USING CROSSOVER
        random.shuffle(population)
        for i in range(0, len(population), 2):
            child1 = ordered_crossover(population[i], population[i+1])
            new_generation.append(child1)
            for _ in range(children_size):
                new_generation.append(tsp_permute(child1))

            child2 = ordered_crossover(population[i+1], population[i])
            new_generation.append(child2)
            for _ in range(children_size):
                new_generation.append(tsp_permute(child2))
        population = new_generation

        # ONLY PERMUTATIONS
        # for solution in population:
        #     for _ in range(children_size):
        #         new_generation.append(tsp_permute(solution))

        if keepParents:
            population += new_generation
        else:
            population = new_generation

        # print(initial_solution in population)
        print(f"Best solution distance {fitness_function(find_best_solution(population))}")

    return find_best_solution(population)

def find_best_solutions(population : list, tournament_size : int) -> list:
    population = sorted(population, key=lambda x: fitness_function(x))
    return population[:tournament_size]

def find_best_solution(population : list) -> list:
    return min(population, key=lambda x: fitness_function(x))

if __name__ == "__main__":
    load_data("input_data/Probleme_Cholet_1_bis")
    # initial_solution = data["init_sol_Cholet_pb1_bis.pickle"]
    initial_solution = [0] + random.sample(range(1, 232), 231) + [232]

    print(f"Initial solution distance {fitness_function(initial_solution)}")
    population_size = 300
    max_generations = 1000
    tournament_size = 50
    children_size = population_size//tournament_size - 1
    keepParents = False
    solution = genetic_algorithm(initial_solution, population_size, max_generations, tournament_size, children_size, keepParents)
    print(f"Final solution distance {fitness_function(solution)}")


