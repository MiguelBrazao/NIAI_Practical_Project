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


def random_search(population_size=1, generations=100, sigma=0.1):
    """
    Optimize the MLP using Random Search.
    """
    agent = MLPAgent

    param_vector = MLPAgent().get_param_vector()
    num_params = len(param_vector)
    best_params = param_vector
    best_reward = -np.inf
    best_rewards = []
    mean_rewards = []

    for generation in range(generations):
        print(f"\n--- Iteration {generation+1}/{generations} ---")
        population = [param_vector + sigma * np.random.randn(num_params) for _ in range(population_size)]
        #with timer_context('Evaluate Parallel'):
        rewards = evaluate_population(agent, population)
        new_population = []

        max_reward_idx = np.argmax(rewards)
        if rewards[max_reward_idx] > best_reward:
            best_reward = rewards[max_reward_idx]
            best_params = deepcopy(population[max_reward_idx])
            # Ensure the directory exists
            Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
            with open(f'data/mlp_best_agents/random_search_seed_{sys.argv[1]}_{best_reward:.3f}.pkl', 'wb') as f:
                pkl.dump(best_params, f)

        # Logging
        print(f"Iteration {generation + 1}: Best Reward = {rewards.max()} Mean Reward = {rewards.mean()}")
        best_rewards.append(rewards.max())
        mean_rewards.append(rewards.mean())
    make_evolution_plot(best_rewards, mean_rewards, "RS", True)
    

    return best_params



if __name__ == "__main__":
    #genetic_algorithm()
    np.random.seed(int(sys.argv[1]))
    torch.random.manual_seed(int(sys.argv[1]))
    random_search()
    
