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
    

def tournament(population, population_size, tournament_k_ratio, rewards, replace=False):
    eligible = np.where(np.isfinite(rewards))[0]
    pool = eligible if len(eligible) >= 2 else np.arange(population_size)
    idx = np.random.choice(pool, min(int(population_size * tournament_k_ratio), len(pool)), replace=replace)
    return deepcopy(population[idx[np.argmax(rewards[idx])]])


def save_best_agent(params, reward):
    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
    inst = MLPAgent()
    inst.set_param_vector(params) # update mlp weights first
    mode = 'hunter' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'move_forward'
    with open(f'data/mlp_best_agents/ga_{mode}_seed_{sys.argv[1]}_{reward:.3f}.pkl', 'wb') as f:
        pkl.dump(inst.get_param_vector(), f) # save the actual agent param vector
    print(f"\nSaved new best agent with reward {reward:.3f}")


def debug_evaluate_population(rewards, population_size, tournament_k_ratio, elite_ratio):
    top_size = max(1, int(population_size * tournament_k_ratio))
    top = np.argsort(rewards)[::-1][:top_size]
    print("\nMax = {:.3f}".format(rewards.max()))
    print("Min = {:.3f}".format(rewards.min()))
    print("Mean = {:.3f}".format(rewards.mean()))
    print("Std = {:.3f}".format(rewards.std()))
    print(f"Top {len(top)}:", top)
    print(f"Top {min(5, len(top))} rewards:")
    for i in range(min(5, len(top))):
        print(f"{i+1}: ", rewards[top[i]])
    print(f"Elite Count: {max(1, int(elite_ratio * population_size))}")


def update_sigma_stagnation(
        new_best_found, current_sigma, stagnation_count,
        population, rewards, num_params,
        sigma, sigma_decay, sigma_min, stagnation_limit, population_size,
        stagnation_ratio, generations, population_restart_ratio
    ):
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
        n_reinit = int(population_size * population_restart_ratio)
        worst_idx = np.argsort(rewards)[:n_reinit]
        for i in worst_idx:
            population[i] = np.random.randn(num_params)
            rewards[i] = -float('inf')  # exclude reinitialized from next tournament
        current_sigma = sigma  # reset to initial sigma on stagnation restart
        stagnation_count = 0
        print(f"\nStagnation Restart")
        print(f"\nReinitialized {n_reinit} individuals")
        print(f"Sigma boosted to {current_sigma:.4f}")
        print(f"\n---------------------------------------------")

    print(f"\nSigma: {current_sigma:.4f}")
    print(f"Stagnation: {stagnation_count}/{stagnation_limit} ({stagnation_ratio*100:.1f}% of {generations} gens)")

    return current_sigma, stagnation_count, population, rewards


def genetic_algorithm(
        generations=300, population_size=60, tournament_k_ratio=0.07, elite_ratio=.05, 
        crossover_rate=0.95, crossover_mask_prob=0.5, mutation_rate=0.1, 
        sigma=0.5, sigma_decay=0.99, sigma_min=0.1,
        stagnation_ratio=0.05, population_restart_ratio=0.75
    ):
    """
    Genetic Algorithm with Tournament Selection, Uniform Crossover, Gaussian Mutation, and Elitism.
    
    Parameters:
    - generations:              how many iterations of the evolutionary process to run.
    - population_size:          the number of candidate solutions (MLP parameter vectors) in each generation.
    - tournament_k_ratio:       the fraction of the population to include in each tournament.
    - elite_ratio:              the fraction of top individuals to carry over unchanged.
    - crossover_rate:           the probability of performing crossover.
    - crossover_mask_prob:      the per-gene probability of taking a gene from parent1 during crossover.
    - mutation_rate:            the probability of mutating each parameter.
    - sigma:                    the mutation strength.
    - sigma_decay:              multiplicative decay applied to sigma each generation (e.g. 0.95 → halves in ~14 gens).
    - sigma_min:                floor for sigma decay so mutation never fully stops.
    - stagnation_ratio:         fraction of total generations without improvement before reinitializing bottom 
                                part of population (e.g. 0.05 with 100 gens → triggers after 5 stagnant gens; 
                                clamped to at least 1).
    - population_restart_ratio: fraction of population to reinitialize upon stagnation 
                                (e.g. 0.5 → reinit bottom half).
                        
    Evolves MLP weights using a Genetic Algorithm with:
        - Tournament selection (parent selection)
        - Uniform crossover
        - Gaussian mutation
        - Elitism (survivor selection)
    """
    agent_class = MLPAgent  # keep class for evaluate_population (used by workers)

    task_label = 'HUNTER' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'MOVE FORWARD'
    stagnation_limit = max(1, round(stagnation_ratio * generations))  # convert ratio to absolute generation count

    print(f"\n---------------------------------------------")
    print(f"\n{task_label} GA Search with MLP Agent")
    print(f"\n---------------------------------------------\n")
    print(f"Generations: {generations}")
    print(f"Population Size: {population_size}")
    print(f"Tournament Top K Ratio: {tournament_k_ratio}")
    print(f"Elite Ratio: {elite_ratio}")
    print(f"Crossover Rate: {crossover_rate}")
    print(f"Crossover Mask Prob: {crossover_mask_prob}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Sigma: {sigma}")
    print(f"Sigma Decay: {sigma_decay}")
    print(f"Sigma Min: {sigma_min}")
    print(f"Stagnation Ratio: {stagnation_ratio}")
    print(f"Population Restart Ratio: {population_restart_ratio}")
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
    
    debug_evaluate_population(rewards, population_size, tournament_k_ratio, elite_ratio) # Print diagnostics about the rewards distribution in the initial population
    save_best_agent(best_params, best_reward) # Save the best agent from the initial population
    new_best_found = True # Flag to track if a new best agent was found in the current generation (used for saving)
    best_rewards = [best_reward] # Track the best reward of each generation for plotting
    mean_rewards = [rewards.mean()] # Track the mean reward of each generation for plotting

    stagnation_count = 0   # generations since last improvement
    current_sigma = sigma  # sigma decays each generation toward sigma_min
    
    # Sigma decay + stagnation restart
    current_sigma, stagnation_count, population, rewards = update_sigma_stagnation(
        new_best_found, current_sigma, stagnation_count,
        population, rewards, num_params,
        sigma, sigma_decay, sigma_min, stagnation_limit, population_size,
        stagnation_ratio, generations, population_restart_ratio
    )
    new_best_found = False
    
    # Main evolutionary loop
    for generation in range(2, generations + 1):
        print(f"\n---------------------------------------------")
        print(f"\nGeneration {generation}/{generations}")

        # Select elites from current population (based on current rewards)
        elite_count = max(1, int(elite_ratio * population_size))
        elite_idx = np.argsort(rewards)[::-1][:elite_count]
        elites = [deepcopy(population[i]) for i in elite_idx]
        elite_rewards_carried = rewards[elite_idx].copy()  # carry forward without re-evaluation
        best_candidate = deepcopy(population[elite_idx[0]])
        best_candidate_reward = rewards[elite_idx[0]]

        # Build new population by reproduction (parents sampled w.r.t current rewards)
        new_population = []
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament(population, population_size, tournament_k_ratio, rewards)
            parent2 = tournament(population, population_size, tournament_k_ratio, rewards)

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

        # Advance to new population; evaluate only offspring, carry elite rewards forward
        population = new_population
        offspring_rewards = evaluate_population(agent_class, population[elite_count:])
        rewards = np.concatenate([elite_rewards_carried, offspring_rewards])
        debug_evaluate_population(rewards, population_size, tournament_k_ratio, elite_ratio)

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

        # Snapshot best/mean BEFORE stagnation update: the restart sets some rewards
        # to -inf which would corrupt rewards.mean() and rewards.max() for the plot.
        best_rewards.append(rewards.max())
        mean_rewards.append(rewards.mean())

        # Sigma decay + stagnation restart
        current_sigma, stagnation_count, population, rewards = update_sigma_stagnation(
            new_best_found, current_sigma, stagnation_count,
            population, rewards, num_params,
            sigma, sigma_decay, sigma_min, stagnation_limit, population_size,
            stagnation_ratio, generations, population_restart_ratio
        )
        new_best_found = False

    make_evolution_plot(best_rewards, mean_rewards, "GA", True)
    inst = MLPAgent()
    inst.set_param_vector(best_params)
    return best_params


if __name__ == "__main__":
    np.random.seed(int(sys.argv[1]))
    torch.random.manual_seed(int(sys.argv[1]))
    genetic_algorithm()