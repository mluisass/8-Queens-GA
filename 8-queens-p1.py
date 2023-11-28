"""
- Representação (genótipo): Permutação de string de bits.
  - Cada indivíduo será uma lista de 8 strings de 8 bits
  - Cada string de 8 bits representa uma linha do tabuleiro
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
import random

TOTAL_RUN_TIMES = 30
POPULATION_SIZE = 100
NUM_CHILDREN = 2
NUM_PARENTS = 2
SELECTION_SIZE = 5
MAX_NUM_ITERATIONS = 10000
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.4


def initialize_population():
    permutation = [format(2**i, '08b') for i in range(8)]
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
            if abs(individual[i].find('1') - individual[j].find('1')) == offset:
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
    return children

def mutation(population):
    # Escolhe dois genes aleatórios e troca os valores
    # ! Não sei se a probabilidade devia entrar aqui por indivíduo ou lá fora (por iteração)
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

def find_solution(population):
    num_iterations = 0
    while num_iterations < MAX_NUM_ITERATIONS:
        parents = parents_selection(population)
        children = parents
        if random.random() < CROSSOVER_PROBABILITY:
            children = crossover(parents)
        children = mutation(children)
        population.extend(children)
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
