import numpy as np
import torch
from agents.mlp_agent import MLPAgent
import sys
import pickle as pkl
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from contextlib import contextmanager
from evaluation import evaluate_population
from pathlib import Path


@contextmanager
def timer_context(label):
    start = time.perf_counter()
    try:
        # Yields control back to the code inside the 'with' block
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}] Elapsed time: {end - start:.4f} seconds")


def make_evolution_plot(best, mean, title, save=False):
    plt.plot(best, label='Best Reward')
    plt.plot(mean, label='Mean Reward')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.draw()
    if save:
        plt.savefig(f'{title}.png')
    plt.pause(0.01)
    plt.clf()
    

# Tournament parent selection
def tournament(population, population_size, tournament_k, rewards, replace=False):
    idx = np.random.choice(population_size, tournament_k, replace=replace)
    return deepcopy(population[idx[np.argmax(rewards[idx])]])


# Genetic Algorithm with Tournament Selection, Uniform Crossover, Gaussian Mutation, and Elitism.
# Population_size is the number of candidate solutions (MLP parameter vectors) in each generation: currently set to 100, which means we evaluate 100 different MLP parameter sets each generation.
# Generations is how many iterations of the evolutionary process to run: currently set to 200, which means we will evolve the population for 200 generations.
# Tournament_k is the number of individuals in each tournament: currently set to 10, which means we randomly select 10 individuals and pick the best among them as a parent.
# Crossover_rate is the probability of performing crossover: currently set to 0.7, which means that 70% of the time we will create a child by combining two parents, and 30% of the time we will just copy one parent (no crossover).
# Sigma is the mutation strength: currently set to 0.2, which means mutations will add noise with std dev of 0.2 to the parameters.
# Mutation_rate is the probability of mutating each parameter: currently set to 0.1, which means each parameter has a 10% chance of being mutated.
# Elite_count is how many top individuals to carry over unchanged: currently set to 4, which means the best 4 individuals from each generation will be directly copied to the next generation without any mutation or crossover.
def genetic_algorithm(population_size=100, generations=200, 
                      tournament_k=15, crossover_rate=0.7, sigma=0.2, mutation_rate=0.1, elite_count=15):
    """
    Evolve MLP weights using a Genetic Algorithm with:
    - Tournament selection (parent selection)
    - Uniform crossover
    - Gaussian mutation
    - Elitism (survivor selection)
    """
    agent = MLPAgent
    num_params = len(MLPAgent().get_param_vector())

    # Initialization
    # We tried using sigma to insert noise
    population = [np.random.randn(num_params) for _ in range(population_size)]

    best_params = population[0]
    best_reward = -np.inf
    best_rewards = []
    mean_rewards = []

    for generation in range(generations):
        print(f"\n--- Generation {generation+1}/{generations} ---")

        # Evaluate
        rewards = evaluate_population(agent, population)

        # Elitism: carry best individual(s) unchanged
        elite_idx = np.argsort(rewards)[::-1][:elite_count]
        new_population = [deepcopy(population[i]) for i in elite_idx]

        # Track global best
        if rewards[elite_idx[0]] > best_reward:
            best_reward = rewards[elite_idx[0]]
            best_params = deepcopy(population[elite_idx[0]])
            Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
            with open(f'data/mlp_best_agents/ga_seed_{sys.argv[1]}_{best_reward:.3f}.pkl', 'wb') as f:
                pkl.dump(best_params, f)

        # Fill rest of next generation
        while len(new_population) < population_size:
            parent1 = tournament(population, population_size, tournament_k, rewards) # Tournament selection for parent 1
            parent2 = tournament(population, population_size, tournament_k, rewards) # Tournament selection for parent 2 (can be the same as parent 1)

            # Crossover
            if np.random.rand() < crossover_rate:
                mask = np.random.rand(num_params) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = deepcopy(parent1)

            # Mutation
            mask = np.random.rand(num_params) < mutation_rate
            child[mask] += sigma * np.random.randn(mask.sum())

            new_population.append(child)

        population = new_population

        print(f"Generation {generation+1}: Best = {rewards.max():.3f}  Mean = {rewards.mean():.3f}")
        best_rewards.append(rewards.max())
        mean_rewards.append(rewards.mean())

    make_evolution_plot(best_rewards, mean_rewards, "GA", True)
    return best_params


if __name__ == "__main__":
    np.random.seed(int(sys.argv[1]))
    torch.random.manual_seed(int(sys.argv[1]))
    genetic_algorithm()