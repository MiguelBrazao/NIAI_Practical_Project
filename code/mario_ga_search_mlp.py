import numpy as np
import torch
from agents.mlp_agent import MLPAgent
import sys
import pickle as pkl
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from contextlib import contextmanager
from evaluation import evaluate_population, TASK_TO_SOLVE
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


def save_best_agent(params, reward):
    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
    inst = MLPAgent()
    inst.set_param_vector(params) # update mlp weights first
    mode = 'hunter' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'move_forward'
    with open(f'data/mlp_best_agents/ga_{mode}_seed_{sys.argv[1]}_{reward:.3f}.pkl', 'wb') as f:
        pkl.dump(inst.get_param_vector(), f) # save the actual agent param vector
    print(f"\nSaved new best agent with reward {reward:.3f}")


def debug_evaluate_population(rewards):
    # DIAGNOSTICS
    top5 = np.argsort(rewards)[::-1][:5]
    print("\nMin = {:.3f}".format(rewards.min()))
    print("Max = {:.3f}".format(rewards.max()))
    print("Mean = {:.3f}".format(rewards.mean()))
    print("Std = {:.3f}".format(rewards.std()))
    print("Top 5:", top5)
    print("Top 5 rewards:")
    print("1: ", rewards[top5[0]])
    print("2: ", rewards[top5[1]])
    print("3: ", rewards[top5[2]])
    print("4: ", rewards[top5[3]])
    print("5: ", rewards[top5[4]])


def update_sigma_stagnation(
        new_best_found, current_sigma, stagnation_count,
        population, rewards, num_params,
        sigma, sigma_decay, sigma_min, stagnation_limit, population_size,
        stagnation_ratio, generations):
    """
    Updates sigma and stagnation state after each generation.

    - Decays current_sigma toward sigma_min each call.
    - Resets stagnation_count to 0 when a new best was found; increments otherwise.
    - When stagnation_count reaches stagnation_limit, reinitializes the worst half of
      the population with fresh random individuals and boosts sigma to re-explore.

    Returns:
        current_sigma (float), stagnation_count (int), population (list), rewards (np.ndarray)
    """
    # Sigma decay
    current_sigma = max(current_sigma * sigma_decay, sigma_min)

    # Stagnation counter
    if new_best_found:
        stagnation_count = 0
    else:
        stagnation_count += 1

    print("\n---------------------------------------------")

    # Stagnation restart
    if stagnation_count >= stagnation_limit:
        n_reinit = population_size // 2
        worst_idx = np.argsort(rewards)[:n_reinit]
        for i in worst_idx:
            population[i] = np.random.randn(num_params)
            rewards[i] = -float('inf')  # exclude reinitialized from next tournament
        current_sigma = min(sigma, current_sigma * 2.0)  # boost sigma to re-explore
        stagnation_count = 0
        print(f"\nStagnation Restart")
        print(f"\nReinitialized {n_reinit} individuals")
        print(f"Sigma boosted to {current_sigma:.4f}")
        print(f"\n---------------------------------------------")

    print(f"\nSigma: {current_sigma:.4f}")
    print(f"Stagnation: {stagnation_count}/{stagnation_limit} ({stagnation_ratio*100:.1f}% of {generations} gens)")

    return current_sigma, stagnation_count, population, rewards


def genetic_algorithm(
                    population_size=40, generations=100, tournament_k=4, 
                    crossover_rate=0.8, sigma=0.7, mutation_rate=0.7, 
                    elite_count=1, crossover_mask_prob=0.5,
                    stagnation_ratio=0.05, sigma_decay=0.95, sigma_min=0.1):
    """
    Genetic Algorithm with Tournament Selection, Uniform Crossover, Gaussian Mutation, and Elitism.
    
    Parameters:
    - Population_size: the number of candidate solutions (MLP parameter vectors) in each generation.
    - Generations: how many iterations of the evolutionary process to run.
    - Tournament_k: the number of individuals in each tournament.
    - Crossover_rate: the probability of performing crossover.
    - Sigma: the mutation strength.
    - Mutation_rate: the probability of mutating each parameter.
    - Elite_count: how many top individuals to carry over unchanged.
    - Crossover_mask_prob: the per-gene probability of taking a gene from parent1 during crossover.
    - Stagnation_ratio: fraction of total generations without improvement before reinitializing bottom half of population (e.g. 0.05 with 100 gens → triggers after 5 stagnant gens; clamped to at least 1).
    - Sigma_decay: multiplicative decay applied to sigma each generation (e.g. 0.95 → halves in ~14 gens).
    - Sigma_min: floor for sigma decay so mutation never fully stops.

    Evolves MLP weights using a Genetic Algorithm with:
        - Tournament selection (parent selection)
        - Uniform crossover
        - Gaussian mutation
        - Elitism (survivor selection)

    Hyperparameter examples:
        - Quick debug (fast, single-process, low cost):
            genetic_algorithm(
                            population_size=20, generations=10, tournament_k=2,
                            crossover_rate=0.8, sigma=0.1, mutation_rate=0.2,
                            elite_count=1, crossover_mask_prob=0.5)

        - Regular debug (fast, single-process, more stable):
            genetic_algorithm(
                            population_size=40, generations=100, tournament_k=3, 
                            crossover_rate=0.8, sigma=0.1, mutation_rate=0.2, 
                            elite_count=4, crossover_mask_prob=0.5)

        - Baseline (reasonable tradeoff):
            genetic_algorithm(
                            population_size=80, generations=100, tournament_k=3,
                            crossover_rate=0.8, sigma=0.25, mutation_rate=0.3,
                            elite_count=2, crossover_mask_prob=0.5)

        - Thorough (more compute, slower convergence but better search):
            genetic_algorithm(
                            population_size=200, generations=300, tournament_k=4,
                            crossover_rate=0.9, sigma=0.15, mutation_rate=0.2,
                            elite_count=4, crossover_mask_prob=0.6)

    Extra notes:
        - For debugging use processes=1 in evaluation.py and small population/generations;
        - Use sigma ∈ [0.1,0.3] for stable searches; higher = noisy exploration;
        - mutation_rate ~0.2–0.4; tournament_k controls selection pressure (higher → stronger pressure);
        - Keep elite_count small (1–5);
        - Fix random seed for repeatability.
    """
    agent_class = MLPAgent  # keep class for evaluate_population (used by workers)

    task_label = 'HUNTER' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'MOVE FORWARD'
    stagnation_limit = max(1, round(stagnation_ratio * generations))  # convert ratio to absolute generation count

    print(f"\n---------------------------------------------")
    print(f"\n{task_label} GA Search with MLP Agent")
    print(f"\n---------------------------------------------\n")
    print(f"Population Size: {population_size}")
    print(f"Generations: {generations}")
    print(f"Tournament Top K: {tournament_k}")
    print(f"Crossover Rate: {crossover_rate}")
    print(f"Sigma: {sigma}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Elite Count: {elite_count}")
    print(f"Crossover Mask Prob: {crossover_mask_prob}")
    print(f"Stagnation Ratio: {stagnation_ratio}")
    print(f"Sigma Decay: {sigma_decay}")
    print(f"Sigma Min: {sigma_min}")
    print(f"\n---------------------------------------------")
    print(f"\nStagnation Limit is {stagnation_limit} generations")
    print(f"\n---------------------------------------------")
    print(f"\nGeneration 1/{generations}")

    # Initialize population with random parameter vectors    
    num_params = len(MLPAgent().get_param_vector())  # get number of parameters from a fresh agent instance
    population = [np.random.randn(num_params) for _ in range(population_size)] # initialize population with random parameter vectors
    rewards = evaluate_population(agent_class, population) # Evaluate all individuals in the population and get their rewards
    best_params = deepcopy(population[np.argmax(rewards)]) # Keep track of the best parameters found so far
    best_reward = rewards.max() # Keep track of the best reward found so far
    
    debug_evaluate_population(rewards) # Print diagnostics about the rewards distribution in the initial population
    save_best_agent(best_params, best_reward) # Save the best agent from the initial population
    new_best_found = True # Flag to track if a new best agent was found in the current generation (used for saving)

    stagnation_count = 0   # generations since last improvement
    current_sigma = sigma  # sigma decays each generation toward sigma_min
    
    # Sigma decay + stagnation restart
    current_sigma, stagnation_count, population, rewards = update_sigma_stagnation(
        new_best_found, current_sigma, stagnation_count,
        population, rewards, num_params,
        sigma, sigma_decay, sigma_min, stagnation_limit, population_size,
        stagnation_ratio, generations
    )
    new_best_found = False

    best_rewards = [best_reward] # Track the best reward of each generation for plotting
    mean_rewards = [rewards.mean()] # Track the mean reward of each generation for plotting
    
    # Main evolutionary loop
    for generation in range(2, generations + 1):
        print(f"\n---------------------------------------------")
        print(f"\nGeneration {generation}/{generations}")

        # Select elites from current population (based on current rewards)
        elite_idx = np.argsort(rewards)[::-1][:elite_count]
        elites = [deepcopy(population[i]) for i in elite_idx]
        best_candidate = deepcopy(population[elite_idx[0]])
        best_candidate_reward = rewards[elite_idx[0]]

        # Build new population by reproduction (parents sampled w.r.t current rewards)
        new_population = []
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament(population, population_size, tournament_k, rewards)
            parent2 = tournament(population, population_size, tournament_k, rewards)

            # Uniform Crossover
            if np.random.rand() < crossover_rate:
                # per-gene probability to take gene from parent1 (default 0.5 = uniform crossover)
                mask = np.random.rand(num_params) < crossover_mask_prob
                child = np.where(mask, parent1, parent2)
            else:
                child = deepcopy(parent1)

            # Gaussian Mutation
            mask = np.random.rand(num_params) < mutation_rate
            child[mask] += current_sigma * np.random.randn(mask.sum())

            new_population.append(child)

        # Apply elitism now (copy elites into new population)
        # Replace the first positions with elites (keeps population size)
        for i, e in enumerate(elites):
            new_population[i] = deepcopy(e)

        # Advance to new population and evaluate it
        population = new_population
        rewards = evaluate_population(agent_class, population)
        debug_evaluate_population(rewards) # Print diagnostics about the rewards distribution in the new population

        # Track global best: check both the old elite (carried forward) and the new population's best
        new_pop_best_idx = np.argmax(rewards)
        new_pop_best_reward = rewards[new_pop_best_idx]
        if best_candidate_reward > best_reward:
            best_reward = best_candidate_reward
            best_params = best_candidate
            new_best_found = True
        if new_pop_best_reward > best_reward:
            best_reward = new_pop_best_reward
            best_params = deepcopy(population[new_pop_best_idx])
            new_best_found = True

        # Save the best params only AFTER weights have been applied to an agent (and after offspring generation)
        if new_best_found:
            save_best_agent(best_params, best_reward)

        # Sigma decay + stagnation restart
        current_sigma, stagnation_count, population, rewards = update_sigma_stagnation(
            new_best_found, current_sigma, stagnation_count,
            population, rewards, num_params,
            sigma, sigma_decay, sigma_min, stagnation_limit, population_size,
            stagnation_ratio, generations
        )
        new_best_found = False

        best_rewards.append(rewards.max())
        mean_rewards.append(rewards.mean())

    make_evolution_plot(best_rewards, mean_rewards, "GA", True)
    inst = MLPAgent()
    inst.set_param_vector(best_params)
    return best_params


if __name__ == "__main__":
    np.random.seed(int(sys.argv[1]))
    torch.random.manual_seed(int(sys.argv[1]))
    genetic_algorithm()