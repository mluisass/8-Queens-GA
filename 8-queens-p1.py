"""
- Representação (genótipo): Permutação de string de bits.
  - Cada indivíduo será uma lista de 8 strings de 3 bits.
  - Cada string de 3 bits representa um número de 1 a 8 (coluna da rainha de uma linha).
- Recombinação: “cut-and-crossfill” crossover
- Probabilidade de Recombinação: 90%
- Mutação: troca de genes
  - Escolher dois genes e trocá-los
- Probabilidade de Mutação: 40%
- Seleção de pais: ranking - Melhor de 2 de 5 escolhidos aleatoriamente
- Seleção de sobreviventes: substituição do pior
- Tamanho da população: 100
- Número de filhos gerados: 2
- Inicialização: aleatória
- Condição de término: Encontrar a solução, ou 10.000
avaliações de fitness
- Fitness
  - Número de pares de rainhas que não se atacam
"""

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
from utils.plot import plot_graph

TOTAL_RUN_TIMES = 30
POPULATION_SIZE = 100
NUM_CHILDREN = 2
NUM_PARENTS = 2
SELECTION_SIZE = 5
MAX_NUM_ITERATIONS = 10000
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.4


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
    # TODO: “cut-and-crossfill”, garantir que filho é uma permutação
    children = []
    cross_point = random.randint(1, 7)
    
    par1_cross = parents[1][cross_point:] + parents[1][:cross_point]
    child1 = parents[0][:cross_point]
    for gene in par1_cross:
        if gene not in child1:
            child1.append(gene)
    
    child2 = parents[1][:cross_point]
    par0_cross = parents[0][cross_point:] + parents[0][:cross_point]
    for gene in par0_cross:
        if gene not in child2:
            child2.append(gene)

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
            return True
    return False

def find_solution(population, analyze):
    num_iterations = 0
    iteration_converged = 0
    fitness_mean = []
    best_fitness = []
    while num_iterations < MAX_NUM_ITERATIONS:
        parents = parents_selection(population)
        children = parents
        if random.random() < CROSSOVER_PROBABILITY:
            children = crossover(parents)
        children = mutation(children)
        population.extend(children)
        population = survival_selection(population)
        if(analyze):
            _, fit_mean, best_fit = analyze_solution(population)
            fitness_mean.append(fit_mean)
            best_fitness.append(best_fit)
        num_iterations += 1
        if is_solution(population):
            break
    return population, num_iterations, iteration_converged, fitness_mean, best_fitness

def analyze_solution(population):
    # Calcula o fitness médio, max e mínimo
    # ! Talvez dê pra fazer isso durante a execução pra plotar no final
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

if __name__ == '__main__':
  converged_iterations_count = 0
  num_iterations_list = []
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

