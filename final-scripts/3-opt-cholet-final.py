import os
import sys
import pickle
import numpy as np
from itertools import combinations
import multiprocessing
import time
import cpuinfo

"""
Ce fichier contient l'implémentation de l'algorithme 3-opt pour le problème du Pays de Cholet.
Auteurs: JANKE Nico, LEGUE Denis
Date: 24/05/2024
Dépendances à installer avec pip: numpy, pickle, cpuinfo
"""

# Constantes
INFINITY = sys.maxsize
data: dict = {}
# Nombre de processus qui vont réaliser les opérations 3-opt en parallèle
num_processes = 24 
# Limite de temps en secondes pour l'exécution de l'algorithme
time_limit = 600 
# np.random.randint(0, 1000000) Seed pour la génération de nombres aléatoires utilisée pour mélanger les segments
random_seed = 322796  
# Si True, l'écran sera effacé à chaque nouvelle itération
do_screen_clear = True
# Nombre d'itérations sans amélioration de la distance avant de revenir à la solution précédente
iteration_without_improvement_threshold = 3
# Nombre d'itérations par processus avant de mettre à jour la solution principale
iterations_per_process = 5


def clear_screen():
    """
    Efface l'écran du terminal.
    """
    print("\033c", end="")

def print_separation():
    """
    Affiche une séparation horizontale.
    """
    print()
    print(os.get_terminal_size()[0] * "=")
    print()

def print_new_iteration(iteration_count):
    """
    Affiche un message indiquant le début d'une nouvelle itération.
    Args:
        iteration_count (int): Le numéro de l'itération.
    """
    if(do_screen_clear):
        print_separation()
    text : str = f" Iteration {iteration_count} "
    separators : str = ((os.get_terminal_size()[0] - len(text)) // 2) * "-"
    full_text : str = separators + text + separators
    if len(full_text) < os.get_terminal_size()[0]:
        full_text = full_text + "-"
    print("\033[93m" + full_text + "\033[00m")


def load_data(folder: str) -> dict:
    """
    Charge les données à partir des fichiers pickle dans le dossier spécifié.
    Les données sont transformées en tableau NumPy et stockées dans un dictionnaire.
    Args:
        folder (str): Le dossier contenant les fichiers pickle.
    Returns:
        dict: Un dictionnaire contenant les données.
    """
    files = os.listdir(folder)
    for file in files:
        with open(folder + "/" + file, 'rb') as f:
            data[file] = np.array(pickle.load(f))  

def save_solution(solution : np.ndarray, best_distance : int, random_seed : int):
    """
    Enregistre la solution ainsi que les paramètres utilisés pour obtenir cette solution dans deux fichiers CSV.
    Args:
        solution (np.ndarray): La solution à enregistrer.
        best_distance (int): La distance de la solution.
        random_seed (int): La seed aléatoire utilisée pour mélanger les segments.
    """
    best_distance = int(best_distance)
    os.makedirs(os.path.dirname("../output_data/"), exist_ok=True)
    os.makedirs(os.path.dirname("../output_data/Problem_Cholet_1_bis/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"../output_data/Problem_Cholet_1_bis/{best_distance}_{random_seed}_{time_limit}_{num_processes}/"), exist_ok=True)
    solution_filename = f"../output_data/Problem_Cholet_1_bis/{best_distance}_{random_seed}_{time_limit}_{num_processes}/solution.csv"
    np.savetxt(solution_filename, solution.flatten(), delimiter=",", fmt="%d", newline=", ")
    params_filename = f"../output_data/Problem_Cholet_1_bis/{best_distance}_{random_seed}_{time_limit}_{num_processes}/parameters.csv"
    with open(params_filename, 'w') as f:
        f.write(f"Best Distance found, {best_distance}\n")
        f.write(f"Random Seed, {random_seed}\n")
        f.write(f"Time Limit, {time_limit}\n")
        f.write(f"Number of Processes, {num_processes}\n")
        cpumodel : str = cpuinfo.get_cpu_info()["brand_raw"]
        f.write(f"CPU Model, {cpumodel}\n") 
        f.write(f"Iteration without improvement threshold, {iteration_without_improvement_threshold}\n") 
        f.write(f"Iterations per process, {iterations_per_process}\n")
        f.close()
    print("\033[92m {}\033[00m" .format("Solution saved"))


def verify_calculate_weight(solution: np.ndarray, data) -> bool:
    """
    Vérifie si la solution respecte les contraintes de poids et renvoie True si c'est le cas, False sinon.
    Args:
        solution (np.ndarray): La solution à vérifier.
        data (dict): Les données du problème.
    Returns:
        bool: True si la solution est valide, False sinon.
    """
    weights = data["weight_Cholet_pb1_bis.pickle"]

    # Assure que les indices de la solution sont des entiers
    solution = solution.astype(int)
    
    # Crée un tableau de poids pour les nœuds de la solution
    node_weights = weights[solution]    

    # Vérifie si la somme cumulée des poids des nœuds à partir de la gauche ne dépasse pas 5850
    cumsum_from_left = np.cumsum(node_weights)
    if np.any(cumsum_from_left > 5850):
        return False

    # Vérifie si la somme cumulée des poids des nœuds à partir de la droite ne dépasse pas 5850
    cumsum_from_right = np.cumsum(node_weights[::-1][1:])
    if np.any(cumsum_from_right > 5850):
        return False
    
    return True

def calculate_total_dist(solution: np.ndarray, data) -> int:
    """
    Calcule la distance totale d'une solution en utilisant la matrice de distance.
    Args:
        solution (np.ndarray): La solution pour laquelle calculer la distance.
        data (dict): Les données du problème.
    Returns:
        int: La distance totale de la solution.
    """
    dist_matrix = data["dist_matrix_Cholet_pb1_bis.pickle"]

    # Assure que les indices de la solution sont des entiers
    solution = solution.astype(int)
    
    # Calcule la distance totale en sommant les distances entre les nœuds consécutifs de la solution
    total_dist = np.sum(dist_matrix[solution[:-1], solution[1:]])  
    return total_dist

def total_distance(solution: np.ndarray, data, best_distance, hasheds) -> int:
    """
    Calcule la distance totale d'une solution et vérifie si elle est valide.
    Si la solution est valide et que sa distance est inférieure à la meilleure distance trouvée, la distance est retournée.
    Sinon, INFINITY est retourné.
    Args:
        solution (np.ndarray): La solution pour laquelle calculer la distance.
        data (dict): Les données du problème.
        best_distance (int): La meilleure distance trouvée jusqu'à présent.
        hasheds (list): Liste des solutions déjà hashées.
    Returns:
        int: La distance totale de la solution si elle est valide et meilleure que la meilleure distance trouvée, sinon INFINITY.
    """
    # Calcule la distance totale de la solution
    distance = calculate_total_dist(solution, data)

    if distance < best_distance:
        if hash(tuple(solution)) not in hasheds: # Vérifie si la solution n'a pas déjà été trouvée
            if verify_calculate_weight(solution, data): # Vérifie si la solution respecte les contraintes de poids
                return distance
    return INFINITY


def get_all_segments(solution):
    """
    Génère tous les segments possibles de la solution.
    Args:
        solution (np.ndarray): La solution pour laquelle générer les segments.
    """
    k = 3
    segments = []
    for indices in combinations(range(1, len(solution) - 1), k):
        if len(set(indices)) == k:
            segments.append(indices)
    return segments


def three_opt_swap(solution, i, j, k):
    """
    Échange le segment [i, j] avec le segment [j, k] de la solution.
    Args:
        solution (np.ndarray): La solution à modifier.
        i (int): L'indice de début du premier segment.
        j (int): L'indice de fin du premier segment et début du deuxième segment.
        k (int): L'indice de fin du deuxième segment.
    """
    new_solution = np.concatenate((solution[:i], solution[j:k], solution[i:j], solution[k:]))
    return new_solution


def three_opt_parallel(segments, best_solution, start_time, shape, solution_lock, event, queue, hashed_solutions, process_number):
    """
    Applique l'opération 3-opt à la solution pour les segments donnés.
    Args:
        segments (list): La liste des segments à traiter.
        best_solution (multiprocessing.Array): La solution à améliorer.
        start_time (float): Le temps de début de l'algorithme.
        shape (tuple): La forme de la solution.
        solution_lock (multiprocessing.Lock): Le verrou pour la solution.
        event (multiprocessing.Event): L'événement pour la synchronisation des processus.
        queue (multiprocessing.Manager().list): L'objet partagé pour envoyer la meilleure solution trouvée à l'updater.
        hashed_solutions (multiprocessing.Manager().list): La liste des solutions déjà hashées.
        process_number (int): Le numéro attribué au processus.
    """
    print("\033[90m {}\033[00m" .format(f"Process {os.getpid()} started"))

    while time.time() - start_time < time_limit:
        # Récupération de la meilleure solution trouvée par le processus et initialisation de la meilleure nouvelle solution 
        with solution_lock:
            best_solution_for_now = np.frombuffer(best_solution, dtype='d').reshape(shape)
        best_new_solution = np.array([])

        # Réalise un certain nombre d'itérations pour les segments donnés. Le nombre d'itérations est défini par la variable iterations_per_process
        for _ in range(iterations_per_process):

            # Initialisation des variables pour stocker la meilleure nouvelle solution et la meilleure nouvelle distance pour l'itération actuelle
            best_iteration_solution = np.array([])
            best_new_distance = INFINITY

            # Boucle du 3-opt avec les segments donnés
            for i, j, k in segments:
                
                # Pour chaque segment, on effectue un swap 3-opt et on calcule la distance de la nouvelle solution. 
                # Le swap est effectué sur la meilleure solution de l'itération précédente si elle existe
                new_solution = three_opt_swap(best_solution_for_now, i, j, k) if best_new_solution.size == 0 else three_opt_swap(best_new_solution, i, j, k)
                new_distance = total_distance(new_solution, data, best_new_distance, hashed_solutions)

                # Si la nouvelle distance est meilleure que la meilleure nouvelle distance, on met à jour la meilleure nouvelle distance et la meilleure nouvelle solution
                if new_distance < best_new_distance:
                    best_new_distance = new_distance
                    best_iteration_solution = new_solution

            # A la fin de l'itération, on met à jour la meilleure solution trouvée pour qu'elle soit utilisée dans la prochaine itération
            best_new_solution = best_iteration_solution

        # La meilleure solution trouvée est ajoutée à la queue pour être traitée par l'updater
        queue[process_number] = (best_new_distance, best_new_solution)

        # On attend que l'updater traite la solution avant de continuer
        event.wait()


def update_solution(best_solution, best_distance, start_time, solution_lock, event, queue, hashed_solutions):
    """
    Met à jour la meilleure solution trouvée par les processus.
    Args:
        best_solution (multiprocessing.Array): La meilleure solution trouvée.
        best_distance (multiprocessing.Value): La meilleure distance trouvée.
        start_time (float): Le temps de début de l'algorithme.
        solution_lock (multiprocessing.Lock): Le verrou pour la solution.
        event (multiprocessing.Event): L'événement pour la synchronisation des processus.
        queue (multiprocessing.Manager().list): L'objet partagé pour recevoir les meilleures solutions trouvées par les processus.
        hashed_solutions (multiprocessing.Manager().list): La liste des solutions déjà hashées.
    """
    print("\033[90m {}\033[00m" .format("Solution updater started"))

    # Initialisation des variables pour stocker les meilleures solutions trouvées et la meilleure solution de tous les temps
    best_solutions_found = []
    best_solution_of_all_time = (INFINITY, None)
    iteration_count = 0
    best_solution_iteration = 0
    iteration_without_improvement = 0

    while time.time() - start_time < time_limit:

        # Si tous les processus ont trouvé une solution, on traite les solutions trouvées
        if(None not in queue):
            iteration_count += 1
            if do_screen_clear:
                clear_screen()
            print_new_iteration(iteration_count)
            print_separation()
            best_queue_solution = (INFINITY, None)

            # On récupère la meilleure solution trouvée par les processus
            for i in range(num_processes):
                queue_solution = queue[i]
                if queue_solution[0] < best_queue_solution[0]:
                    best_queue_solution = queue_solution

            # On réinitialise la queue
            for i in range(num_processes):
                queue[i] = None

            # On vérifie si la meilleure solution trouvée est meilleure que la meilleure solution de tous les temps et on met à jour les variables en conséquence
            if best_queue_solution[0] < best_solution_of_all_time[0]:
                print("\033[92m {}\033[00m" .format("New absolute best solution found with a distance of"), best_queue_solution[0])
                print("\033[92m {}\033[00m" .format("Best solution:"), [int(x) for x in best_queue_solution[1]])
                print()
                best_solution_of_all_time = best_queue_solution
                best_solution_iteration = iteration_count
                
            else:
                print("\033[93m {}\033[00m" .format("No new absolute best solution found"))
                print("\033[93m {}\033[00m" .format("Last iteration that improved the best solution:"), best_solution_iteration)
                print("\033[93m {}\033[00m" .format("Best solution found:"), [int(x) for x in best_solution_of_all_time[1]])
                print("\033[93m {}\033[00m" .format("With distance:"), best_solution_of_all_time[0])
                print()

            with best_distance.get_lock() and solution_lock:

                # On vérifie si la meilleure solution trouvée est meilleure que la meilleure solution de l'itération précédente
                # Si c'est le cas, on met à jour la solution qui sera utilisée dans la prochaine itération et on réinitialise le compteur d'itérations sans amélioration
                # Sinon, on incrémente le compteur d'itérations sans amélioration
                iteration_variation = best_distance.value - best_queue_solution[0]
                if iteration_variation == 0: 
                    iteration_without_improvement += 1 
                else:
                    if iteration_variation > 0:
                        best_solutions_found.append(best_queue_solution)
                    iteration_without_improvement = 0

                print("\033[94m {}\033[00m" .format(f"Previous iteration best distance:"), best_distance.value)
                print("\033[94m {}\033[00m" .format(f"Iteration best distance:"), best_queue_solution[0])
                print("\033[94m {}\033[00m" .format(f"Iteration variation value:"), iteration_variation)
                print("\033[94m {}\033[00m" .format(f"Time:"), time.time() - start_time)

                # Si le nombre d'itérations sans amélioration dépasse le seuil, on revient à la dernière solution trouvée qui avait une autre distance que la solution actuelle
                # S'il n'y a pas d'autre solution, on arrête l'algorithme
                if iteration_without_improvement >= iteration_without_improvement_threshold:
                    if(len(best_solutions_found) > 1): 
                        print()
                        best_solutions_found.pop(-1)
                        print("\033[93m{}\033[00m" .format(f"Going back to a previous solution whit a distance of {best_solutions_found[-1][0]} in order to find a new way to get a better solution"))
                        iteration_without_improvement = 0
                    else:
                        print("\033[91m {}\033[00m" .format("No new way to get a better solution found"))
                        print_separation()
                        best_distance.value = best_solution_of_all_time[0]
                        best_solution[:] = best_solution_of_all_time[1].flatten()
                        return

                # On met à jour la meilleure solution trouvée pour l'itération actuelle
                best_distance.value = best_solutions_found[-1][0]
                best_solution[:] = best_solutions_found[-1][1].flatten()

            # On vérifie si la solution n'a pas déjà été trouvée et on l'ajoute à la liste des solutions déjà hashées
            hashed = hash(tuple(best_queue_solution[1]))
            assert hashed not in hashed_solutions, "\033[91m {}\033[00m" .format("Solution already hashed")
            hashed_solutions.append(hashed)

            print_separation()

            # On signale aux processus que la meilleure solution trouvée a été traitée et que la prochaine itération peut commencer
            event.set()
            event.clear()

    # Lorsque le temps est écoulé, on vérifie si une solution a été trouvée
    if best_solution_of_all_time == (INFINITY, None):
        print("\033[91m {}\033[00m" .format("No solution found"))
        print_separation()
        return

    # Lorque le temps est écoulé, on met à jour la meilleure solution trouvée avec la meilleure solution de tous les temps
    with best_distance.get_lock() and solution_lock:
        best_distance.value = best_solution_of_all_time[0]
        best_solution[:] = best_solution_of_all_time[1].flatten()


def three_opt(solution):
    """
    Applique l'algorithme 3-opt à la solution donnée.
    Args:
        solution (np.ndarray): La solution à améliorer.
    """

    # Début du chronomètre
    start = time.time()
    # Initialisation de la meilleure distance trouvée
    best_distance = total_distance(solution, data, INFINITY, [])
    # Initialisation de la liste des segments
    segments = get_all_segments(solution)

    # Mélange des segments si une seed aléatoire est spécifiée
    if random_seed is not None:
        np.random.seed(random_seed)
        np.random.shuffle(segments)

    # Division des segments en chunks pour les processus
    processes = []
    chunk_size = len(segments) // num_processes
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    if len(segments) % num_processes != 0:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)

    # Initialisation des variables partagées entre les processus
    solution_shape = solution.shape
    best_solution = multiprocessing.Array('d', solution.flatten(), lock=False)
    best_solution_lock = multiprocessing.Lock()
    best_distance_value = multiprocessing.Value('d', best_distance)
    event = multiprocessing.Event()
    hashed_solutions = multiprocessing.Manager().list()
    queue = multiprocessing.Manager().list()

    for _ in range(num_processes):
        queue.append(None)

    # Création des processus
    process_number = 0
    for chunk in chunks:
        p = multiprocessing.Process(target=three_opt_parallel, args=(chunk, best_solution, start, solution_shape, best_solution_lock, event, queue, hashed_solutions, process_number))
        p.start()
        processes.append(p)
        process_number += 1

    # Création du processus updater
    solution_updater = multiprocessing.Process(target=update_solution, args=(best_solution, best_distance_value, start, best_solution_lock, event, queue, hashed_solutions))
    solution_updater.start()
    processes.append(solution_updater)

    # Attente de la fin de l'updater
    solution_updater.join()

    # Arrêt forcé des processus
    for p in processes:
        p.terminate()

    # Récupération de la meilleure solution trouvée
    with best_solution_lock:
        solution_found = np.frombuffer(best_solution, dtype='d').reshape(solution_shape)
    with best_distance_value.get_lock():
        best_distance_found = best_distance_value.value
    
    print("Random seed:", random_seed)

    return solution_found, best_distance_found



if __name__ == "__main__":

    print_separation()
    print("Number of processes:", num_processes)
    load_data("../input_data/Probleme_Cholet_1_bis")
    solution = data["init_sol_Cholet_pb1_bis.pickle"]
    print("Initial solution:", solution)
    print("Initial distance:", total_distance(solution, data, INFINITY, []))
    print_separation()

    best_solution, best_distance = three_opt(solution)

    print_separation()
    print("\033[92m {}\033[00m" .format("Best solution:"), [int(x) for x in best_solution])
    print("\033[92m {}\033[00m" .format("Best distance:"), best_distance)
    print_separation()

    save_solution(best_solution, best_distance, random_seed)


