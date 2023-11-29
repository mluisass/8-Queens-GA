'''
    - TODO: estatísticas
    - Em quantas execuções o algoritmo convergiu das 30; (OK)
    - Em que iteração o algoritmo convergiu (média e desvio padrão); (OK)
    - Número de indivíduos que convergiram por execução; --> printar
    - Fitness médio da população em cada uma das 30 execuções; (OK)
    - Colocar gráficos de convergência com a média e o melhor indivíduo por iteração;
    - Fitness médio alcançado nas 30 execuções (média e desvio padrão); (OK)
    - Análise adicional: Quantas iterações são necessárias para toda a população convergir?

    - Testar trocando seleção por roleta!
'''
import random
import numpy as np
import matplotlib.pyplot as plt

TOTAL_RUN_TIMES = 30
POPULATION_SIZE = 50
NUM_CHILDREN = 2
NUM_PARENTS = 2
SELECTION_SIZE = 5
MAX_NUM_ITERATIONS = 10000
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.3
NUMBER_OF_CROSSOVERS = 10 # Gera 20 filhos por iteração

def initialize_population():
    permutation = [str(bin(i)) for i in range(1,9)]
    population = [random.sample(permutation, 8) for _ in range(POPULATION_SIZE)]
    return population

def fitness(individual):
    # Número de pares de rainhas que não se atacam
    # Máximo = 28 (C(8,2) = 8! / (2! * (8 - 2)!) = 28)
    # Mínimo = 0
    fitness = 28
    for i in range(8):
        for j in range(i + 1, 8):
            if individual[i] == individual[j]: # mesma coluna (duas linhas iguais)
                fitness -= 1
            # mesma diagonal (diferença entre as linhas é igual à diferença entre as colunas)
            offset = j - i
            if abs(int(individual[i], 2) - int(individual[j], 2)) == offset:
                fitness -= 1
            
    return fitness

def parents_selection(population):
    # Pais = 2 melhores de 5
    selected = random.sample(population, SELECTION_SIZE)
    selected.sort(key=lambda individual: fitness(individual), reverse=True)
    return selected[0:NUM_PARENTS]

def survival_selection(population):
    # Sobreviventes = 100 melhores
    population.sort(key=lambda individual: fitness(individual), reverse=True)
    return population[0:POPULATION_SIZE]

def crossover(parents):
    # Crossover com cruzamento de ordem 1
    children = []
    cross_left_point = 2
    cross_right_point = 6
    child1 = [0] * 8
    child1_idx = 6
    child2 = [0] * 8
    child2_idx = 6

    child1[cross_left_point:cross_right_point] = parents[0][cross_left_point:cross_right_point]
    for gene in parents[1]:
        if gene not in child1[cross_left_point:cross_right_point]:
            child1[child1_idx] = gene
            child1_idx = (child1_idx + 1) % 8

    child2[cross_left_point:cross_right_point] = parents[1][cross_left_point:cross_right_point]

    for gene in parents[0]:
        if gene not in child2[cross_left_point:cross_right_point]:
            child2[child2_idx] = gene
            child2_idx = (child2_idx + 1) % 8

    children += [child1, child2]

    return children

def mutation(population):
    # Escolhe dois genes aleatórios e troca os valores
    for individual in population:
        if random.random() < MUTATION_PROBABILITY:
            gene1, gene2 = random.sample(range(8), 2)
            individual[gene1], individual[gene2] = individual[gene2], individual[gene1]
    return population

def is_solution(population):
    for individual in population:
        if fitness(individual) == 28:
            for i in range(8):
                line = [0] * 8
                line[int(individual[i],2)-1] = 1
            return True
    return False

def find_solution(population, analyze):
    num_iterations = 0
    iteration_converged = 0
    fitness_mean = []
    best_fitness = []
    while num_iterations < MAX_NUM_ITERATIONS:
        num_crossovers = 0
        while num_crossovers < NUMBER_OF_CROSSOVERS:
            parents = parents_selection(population)
            children = parents
            if random.random() < CROSSOVER_PROBABILITY:
                children = crossover(parents)
            children = mutation(children)
            population.extend(children)
            num_crossovers += 1
        population = survival_selection(population)
        if (analyze):
            _, fit_mean, best_fit = analyze_solution(population)
            fitness_mean.append(fit_mean)
            best_fitness.append(best_fit)
        num_iterations += 1
        if is_solution(population):
            iteration_converged = 1
            break
    return population, num_iterations, iteration_converged, fitness_mean, best_fitness

def analyze_solution(population):
    # Calcula o fitness médio, max e mínimo
    fitness_sum = 0
    fitness_max = 0
    fitness_min = 28
    converged_individuals = 0

    for individual in population:
        individual_fitness = fitness(individual)
        fitness_sum += individual_fitness
        if individual_fitness > fitness_max:
            fitness_max = individual_fitness
        if individual_fitness < fitness_min:
            fitness_min = individual_fitness
        if individual_fitness == 28:
            converged_individuals += 1

    return converged_individuals, (fitness_sum / POPULATION_SIZE), fitness_max

def plot_graph(y, title, x_label, y_label):
    mean = np.mean(y)
    std = np.std(y)

    plt.figure(figsize=(12, 4))
    plt.plot(y, color='blue', linewidth=2)
    plt.axhline(y=mean, color='r', linestyle='--')
    plt.axhline(y=mean + std, color='r', linestyle='--')
    plt.axhline(y=mean - std, color='r', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
  converged_iterations_count = 0
  num_iterations_list = [] # número de iterações
  converged_per_execution = []
  fit_mean_per_execution = []
  for run_time in range(0, TOTAL_RUN_TIMES):
    initial_population = initialize_population()
    population, num_iterations, iteration_converged, fit_mean, best_fit = find_solution(initial_population, True if run_time == TOTAL_RUN_TIMES-1 else False)
    num_iterations_list.append(num_iterations)
    converged_iterations_count += iteration_converged
    converged_count, fitness_mean, _ = analyze_solution(population)
    converged_per_execution.append(converged_count)
    fit_mean_per_execution.append(fitness_mean)

  print('Número de iterações convergidas: ', converged_iterations_count)

  # Informações sobre iterações e média
  plot_graph(num_iterations_list, 'Número de iterações por execução', 'Execução', 'Número de iterações')

  # Média do fitness por iteração
  plot_graph(fit_mean_per_execution, 'Fitness médio por execução', 'Execução', 'Fitness médio')
  
  # Melhor indivíduo por iteração
  print(best_fit)
  plot_graph(best_fit, 'Melhor indivíduo por iteração', 'Iteração', 'Fitness')

  # Fitness médio por iteração
  print(fit_mean)
  plot_graph(fit_mean, 'Fitness médio por iteração', 'Iteração', 'Fitness médio')


