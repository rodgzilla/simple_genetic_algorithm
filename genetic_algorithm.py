from collections import defaultdict
from functools   import partial

import matplotlib.pyplot    as plt
import matplotlib.animation as animation
import random
import math

def create_point():
    """
    Creates a random point with 0 <= x, y <= 1.
    """
    return random.random(), random.random()

def distance(orig, dest):
    """
    Simple euclidean distance between two points.
    """
    return math.sqrt((dest[0] - orig[0]) ** 2 + (dest[1] - orig[1]) ** 2)

def compute_distance_matrix(point_set):
    """
    Computes the matrix of distance between every couple of point.
    """
    distance_matrix = defaultdict(dict)
    for orig in point_set:
        for dest in point_set:
            distance_matrix[orig][dest] = distance_matrix[dest][orig] = distance(orig, dest)
    return distance_matrix

def compute_solution_distance(solution, distance_matrix):
    """
    Computes the total length of a solution.
    """
    total_distance = 0

    for i in range(len(solution) - 1):
        total_distance += distance_matrix[solution[i]][solution[i + 1]]

    return total_distance

def create_individual(point_set):
    """
    Creates a random new individual.
    """
    points = list(point_set)
    random.shuffle(points)

    return points

def create_population(n_individuals, point_set, distance_matrix):
    """
    Initializes the population.
    """
    individuals = [create_individual(point_set) for _ in
                   range(n_individuals)]
    distances = list(map(partial(compute_solution_distance,
                                 distance_matrix = distance_matrix), individuals))

    return sorted(zip(individuals, distances), key = lambda x: x[1])

def plot_result(solution):
    """
    Plots a solution using matplotlib.pyplot
    """
    xs = [point[0] for point in solution]
    ys = [point[1] for point in solution]
    plt.plot(xs, ys)
    plt.axis('off')
    plt.show()

def plot_point_set(point_set):
    """
    Plots a set of points using matplotlib.pyplot
    """
    point_list = list(point_set)
    xs = [point[0] for point in point_list]
    ys = [point[1] for point in point_list]
    plt.scatter(xs, ys)
    plt.axis('off')
    plt.show()

def mutate(individual, distance_matrix):
    def mutation_swap():
        swap_idx       = random.randint(0, len(individual) - 2)
        new_individual = individual[:swap_idx] + \
                         [individual[swap_idx + 1], individual[swap_idx]] + \
                         individual[swap_idx + 2:]

        return new_individual

    def mutation_reverse():
        reverse_start = random.randint(0, len(individual) - 2)
        reverse_end   = random.randint(reverse_start + 1, len(individual) - 1)
        new_individual = individual[:reverse_start] + \
                         individual[reverse_start : reverse_end][::-1] + \
                         individual[reverse_end:]

        return new_individual

    mutation = random.choice([mutation_swap, mutation_reverse])
    # mutation = mutation_reverse
    new_individual = mutation()

    return new_individual, compute_solution_distance(new_individual, distance_matrix)

def reproduce(individual_1, individual_2, distance_matrix):
    def generate_subset_idx(subset_size):
        return sorted(random.sample(range(ind_size), subset_size))

    def select_subset(individual, subset_idx):
        return [individual[i] for i in subset_idx]

    def complement_subset(individual_2, individual_1_subset):
        s = set(individual_1_subset)

        return [point for point in individual_2 if point not in s]

    ind_size           = len(individual_1)
    ind_1_subset_size = ind_size // 2
    subset_ind_1_idx  = generate_subset_idx(ind_1_subset_size)
    ind_1_subset      = select_subset(individual_1, subset_ind_1_idx)
    ind_2_subset      = complement_subset(individual_2, ind_1_subset)

    new_individual = ind_1_subset + ind_2_subset

    return new_individual, compute_solution_distance(new_individual,
                                                     distance_matrix)

def evolve(population, n_reproductions, n_mutations, n_news,
           reproductor_pool, distance_matrix, point_set):
    """This function computes one step of evolution of the population. It
    selects `n_reproductions` individuals in the top
    `reproduction_pool` part of the list and reproducts them with
    another random individual. `n_mutations` mutations of random
    individual are added to the populations to keep genome diversity.
    Finally `n_news` individual are randomly created and added to the
    population.
    """
    population_size       = len(population)
    n_new_individuals     = n_reproductions + n_mutations + n_news
    n_survivors           = population_size - n_new_individuals
    reproductor_pool_size = round(reproductor_pool * population_size)
    new_population        = population[:n_survivors]

    for _ in range(n_reproductions):
        individual_1 = population[random.randint(0, reproductor_pool_size - 1)][0]
        individual_2 = random.choice(population)[0]
        new_population.append(reproduce(individual_1, individual_2, distance_matrix))

    for _ in range(n_mutations):
        individual_to_mutate = random.choice(population)[0]
        new_population.append(mutate(individual_to_mutate, distance_matrix))

    for _ in range(n_news):
        new_individual = create_individual(point_set)
        new_population.append((new_individual,
                               compute_solution_distance(new_individual, distance_matrix)))

    return sorted(new_population, key = lambda x: x[1])

def genetic_algorithm(point_set, population_size, n_generations,
                      n_reproductions, n_mutations, n_news, reproduction_pool):
    """
    Creates a population and make it evolve according to the arguments.
    """
    distance_matrix = compute_distance_matrix(point_set)
    population      = create_population(population_size, point_set, distance_matrix)

    for i in range(n_generations):
        population = evolve(population, n_reproductions, n_mutations,
                            n_news, reproduction_pool, distance_matrix, point_set)
        print('Generation', i, 'best individual length:', population[0][1])

    return population[0]

if __name__ == '__main__':
    point_set             = {create_point() for _ in range(40)}
    plot_point_set(point_set)
    best_solution, length = genetic_algorithm(point_set, 3000, 1200, 1000, 500, 0, 0.15)
    plot_result(best_solution)
