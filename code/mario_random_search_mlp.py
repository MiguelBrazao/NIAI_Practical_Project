import numpy as np
import torch
from agents.mlp_agent import MLPAgent
import sys
import pickle as pkl
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from evaluation import evaluate_population, TASK_TO_SOLVE
from pathlib import Path


class Tee:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


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


def random_search(population_size=100, generations=250, sigma=0.5, seed=None):
    """
    Optimize the MLP using Random Search.
    """
    agent = MLPAgent

    task_label = 'HUNTER' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'MOVE FORWARD'

    print(f"\n=============================================")
    print(f"\n{task_label} Random Search with MLP Agent")
    print(f"\n=============================================\n")
    print(f"Seed: {seed}")
    print(f"Generations: {generations}")
    print(f"Population Size: {population_size}")
    print(f"Sigma: {sigma}")

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
        rewards, kills, coins, distance, levels = evaluate_population(agent, population)
        new_population = []

        max_reward_idx = np.argmax(rewards)
        if rewards[max_reward_idx] > best_reward:
            best_reward = rewards[max_reward_idx]
            best_params = deepcopy(population[max_reward_idx])
            # Ensure the directory exists
            Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
            with open(f'data/mlp_best_agents/random_search_{task_label.lower().replace(" ", "_")}_seed_{sys.argv[1]}_{best_reward:.3f}.pkl', 'wb') as f:
                pkl.dump(best_params, f)

        # Logging
        best_idx = np.argmax(rewards)
        print(f"Iteration {generation + 1}: Best Reward = {rewards.max():.3f} | Mean Reward = {rewards.mean():.3f}")
        print(f"  Best: Kills={kills[best_idx]}  Coins={coins[best_idx]}  Distance={distance[best_idx]:.1f}  Levels completed={levels[best_idx]}")
        best_rewards.append(rewards.max())
        mean_rewards.append(rewards.mean())
    make_evolution_plot(best_rewards, mean_rewards, "RS", True)
    

    return best_params



if __name__ == "__main__":
    seed = int(sys.argv[1])
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    method = 'hunter' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'move_forward'
    log_dir = Path("data")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rs_{method}_seed_{seed}.txt"

    with open(log_path, 'a', encoding='utf-8') as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            random_search(seed=seed)
    
