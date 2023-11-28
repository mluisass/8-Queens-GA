'''
    - TODO: estatísticas
    - Em quantas execuções o algoritmo convergiu das 30;
    - Em que iteração o algoritmo convergiu (média e desvio padrão);
    - Número de indivíduos que convergiram por execução;
    - Fitness médio da população em cada uma das 30 execuções;
    - Colocar gráficos de convergência com a média e o melhor indivíduo por iteração;
    - Fitness médio alcançado nas 30 execuções (média e desvio padrão);
    - Análise adicional: Quantas iterações são necessárias para toda a população convergir?
'''
import random

TOTAL_RUN_TIMES = 30
POPULATION_SIZE = 100
NUM_CHILDREN = 2
NUM_PARENTS = 2
SELECTION_SIZE = 5
MAX_NUM_ITERATIONS = 10000
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.3
NUMBER_OF_CROSSOVERS = 10

def initialize_population():
    permutation = [str(bin(i)) for i in range(1,9)]
    population = [random.sample(permutation, 8) for _ in range(POPULATION_SIZE)]
    return population

def fitness(individual):
    # Número de pares de rainhas que não se atacam
    # Máximo = 28 (C(8,2) = 8! / (2! * (8 - 2)!) = 28)
    # Mínimo = 0
    # ! Não testei se isso tá certo
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
                print(line)
            print ('\n')
            return True
    return False

def find_solution(population):
    num_iterations = 0
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
        num_iterations += 1
        if is_solution(population):
            break
    return population, num_iterations

def analyze_solution(population, num_iterations):
    # Calcula o fitness médio, max e mínimo
    # ! Talvez dê pra fazer isso durante a execução pra plotar no final
    fitness_sum = 0
    fitness_max = 0
    fitness_min = 28

    for individual in population:
        individual_fitness = fitness(individual)
        fitness_sum += individual_fitness
        if individual_fitness > fitness_max:
            fitness_max = individual_fitness
        if individual_fitness < fitness_min:
            fitness_min = individual_fitness
    print('Fitness médio:', fitness_sum / POPULATION_SIZE)
    print('Fitness máximo:', fitness_max)
    print('Fitness mínimo:', fitness_min)
    print('Número de iterações:', num_iterations)

if __name__ == '__main__':
  for run_time in range(0, TOTAL_RUN_TIMES):
    initial_population = initialize_population()
    population, num_iterations = find_solution(initial_population)
    analyze_solution(population, num_iterations)
