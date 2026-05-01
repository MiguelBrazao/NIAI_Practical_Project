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


def make_evolution_plot(best, mean, save=False):
    mode = 'hunter' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'move_forward'
    plt.plot(best, label='Best Reward')
    plt.plot(mean, label='Mean Reward')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title('Genetic Algorithm Evolution - ' + mode.capitalize() + f' (Best Reward: {best[-1]:.3f})')
    plt.legend()
    plt.draw()
    if save:
        plt.savefig(f'ga_{mode}_seed_{sys.argv[1]}_{best[-1]:.3f}.png')
    plt.pause(0.01)
    plt.clf()
    

def tournament(population, tournament_k, rewards, replace=False):
    eligible = np.where(np.isfinite(rewards))[0] # only consider individuals with valid rewards for tournament selection (exclude -inf from reinitialized individuals)
    pool = eligible if len(eligible) >= 2 else np.arange(len(population)) # if not enough eligible individuals, fallback to entire population to avoid errors
    idx = np.random.choice(pool, min(tournament_k, len(pool)), replace=replace) # select tournament_k individuals from the eligible pool without replacement (or with replacement if not enough eligible)
    return deepcopy(population[idx[np.argmax(rewards[idx])]]) # return a copy of the best individual from the tournament


def save_best_agent(params, reward):
    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
    inst = MLPAgent()
    inst.set_param_vector(params) # update mlp weights first
    mode = 'hunter' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'move_forward'
    with open(f'data/mlp_best_agents/ga_{mode}_seed_{sys.argv[1]}_{reward:.3f}.pkl', 'wb') as f:
        pkl.dump(inst.get_param_vector(), f) # save the actual agent param vector
    print(f"\nSaved new best agent with reward {reward:.3f}")


def debug_evaluate_population(rewards, tournament_k, elite_count, kills=None):
    top = np.argsort(rewards)[::-1][:tournament_k]
    print("\nMax = {:.3f}".format(rewards.max()))
    print("Min = {:.3f}".format(rewards.min()))
    print("Mean = {:.3f}".format(rewards.mean()))
    print("Std = {:.3f}".format(rewards.std()))
    print(f"Top {len(top)}:", top)
    print(f"Top {min(5, len(top))} rewards:")
    for i in range(min(5, len(top))):
        print(f"{i+1}: ", rewards[top[i]])
    print(f"Elite Count: {elite_count}")
    if kills is not None:
        print(f"\nBest Candidate Kills: {kills[np.argmax(rewards)]}")


def update_sigma_stagnation(
        new_best_found, current_sigma, stagnation_count,
        population, rewards, num_params,
        sigma, sigma_decay, sigma_min, sigma_max, stagnation_limit, population_size,
        stagnation_ratio, generations, restart_ratio_min, restart_ratio_max, reference_std
    ):
    """
    Updates sigma and stagnation state after each generation.

    - Decays current_sigma toward sigma_min each call.
    - Resets stagnation_count to 0 when a new best was found; increments otherwise.
    - When stagnation_count reaches stagnation_limit, reinitializes the bottom portion of
      the population with fresh random individuals and boosts sigma to re-explore.
    - The restart ratio is adaptive: low reward std (converged population) → large restart;
      high reward std (diverse population) → small restart.

    Returns:
        current_sigma (float), stagnation_count (int), population (list), rewards (np.ndarray)
    """
    # Decay sigma when improving, grow it back when stagnating (up to sigma_max)
    if new_best_found:
        current_sigma = max(current_sigma * sigma_decay, sigma_min)
    else:
        current_sigma = min(current_sigma / sigma_decay, sigma_max)

    # Stagnation counter
    if new_best_found:
        stagnation_count = 0
    else:
        stagnation_count += 1

    print("\n---------------------------------------------")

    # Stagnation restart
    if stagnation_count >= stagnation_limit:
        # Adaptive restart ratio: low std (converged) → restart more; high std (diverse) → restart less
        finite_rewards = rewards[np.isfinite(rewards)] # consider only finite rewards for std calculation to avoid issues with -inf from reinitialized individuals
        current_std = finite_rewards.std() if len(finite_rewards) > 1 else 0.0 # handle case with 0 or 1 valid rewards
        std_norm = np.clip(current_std / reference_std, 0.0, 1.0) if reference_std > 0 else 0.0 # normalise to [0,1] based on initial generation's std, with fallback to 0 if reference_std is 0 (e.g., all rewards are identical)
        population_restart_ratio = restart_ratio_max - (restart_ratio_max - restart_ratio_min) * std_norm # linear interpolation between min and max restart ratio based on normalised std

        n_reinit = int(population_size * population_restart_ratio) # number of individuals to reinitialize
        worst_idx = np.argsort(rewards)[:n_reinit] # indices of the worst-performing individuals to replace with new random ones
        for i in worst_idx: 
            population[i] = np.random.randn(num_params) # reinitialize with new random parameters
            rewards[i] = -float('inf') # exclude reinitialized from next tournament
        # Adaptive sigma on restart: low std (converged) → sigma_max; high std (diverse) → sigma
        current_sigma = sigma + (sigma_max - sigma) * (1.0 - std_norm) # linear interpolation between sigma and sigma_max based on normalised std
        stagnation_count = 0 # reset stagnation count after restart
        print(f"\nStagnation Restart")
        print(f"Reward Std: {current_std:.1f} (ref: {reference_std:.1f}, norm: {std_norm:.2f})")
        print(f"Restart Ratio: {population_restart_ratio:.2f} → Reinitialized {n_reinit} individuals")
        print(f"Sigma set to {current_sigma:.4f}")
        print(f"\n---------------------------------------------")

    print(f"\nSigma: {current_sigma:.4f}")
    print(f"Stagnation: {stagnation_count}/{stagnation_limit} ({stagnation_ratio*100:.1f}% of {generations} gens)")

    return current_sigma, stagnation_count, population, rewards


def genetic_algorithm(
        generations=250, population_size=100, tournament_k=4, elite_count=3, 
        crossover_rate=0.95, crossover_mask_prob=0.5, mutation_rate=0.1, 
        sigma=0.5, sigma_decay=0.99, sigma_min=0.1, sigma_max=1.5,
        stagnation_ratio=0.1, restart_ratio_min=0.25, restart_ratio_max=0.75
    ):
    """
    Genetic Algorithm with Tournament Selection, Uniform Crossover, Gaussian Mutation, and Elitism.
    
    Parameters:
    - generations:              how many iterations of the evolutionary process to run.
    - population_size:          the number of candidate solutions (MLP parameter vectors) in each generation.
    - tournament_k:             number of individuals to include in each tournament.
    - elite_count:              number of top individuals to carry over unchanged.
    - crossover_rate:           the probability of performing crossover.
    - crossover_mask_prob:      the per-gene probability of taking a gene from parent1 during crossover.
    - mutation_rate:            the probability of mutating each parameter.
    - sigma:                    the initial mutation strength (also the reset value after stagnation restart).
    - sigma_decay:              multiplicative decay applied to sigma each generation (e.g. 0.95 → halves in ~14 gens).
    - sigma_min:                floor for sigma so mutation never fully stops.
    - sigma_max:                ceiling for sigma growth during stagnation (defaults to 3 * sigma if None).
    - stagnation_ratio:         fraction of total generations without improvement before reinitializing bottom 
                                part of population (e.g. 0.05 with 100 gens → triggers after 5 stagnant gens; 
                                clamped to at least 1).
    - restart_ratio_min:        minimum fraction of population to reinitialize (used when reward std is high).
    - restart_ratio_max:        maximum fraction of population to reinitialize (used when reward std is low).
                        
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
    print(f"Tournament K: {tournament_k}")
    print(f"Elite Count: {elite_count}")
    print(f"Crossover Rate: {crossover_rate}")
    print(f"Crossover Mask Prob: {crossover_mask_prob}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Sigma: {sigma}")
    print(f"Sigma Decay: {sigma_decay}")
    print(f"Sigma Min: {sigma_min}")
    print(f"Sigma Max: {sigma_max}")
    print(f"Stagnation Ratio: {stagnation_ratio}")
    print(f"Restart Ratio Min: {restart_ratio_min}")
    print(f"Restart Ratio Max: {restart_ratio_max}")
    print(f"\n---------------------------------------------")
    print(f"\nStagnation Limit is {stagnation_limit} generations")
    print(f"\n---------------------------------------------")
    print(f"\nGeneration 1/{generations}")

    # Initialize population with random parameter vectors    
    num_params = len(MLPAgent().get_param_vector())  # get number of parameters from a fresh agent instance
    population = [np.random.randn(num_params) for _ in range(population_size)] # initialize population with random parameter vectors
    rewards, kills = evaluate_population(agent_class, population) # Evaluate all individuals in the population and get their rewards
    best_params = deepcopy(population[np.argmax(rewards)]) # Keep track of the best parameters found so far
    best_reward = rewards.max() # Keep track of the best reward found so far
    reference_std = rewards.std()  # baseline std from fully random generation 1, used to normalise adaptive restart ratio
    
    debug_evaluate_population(rewards, tournament_k, elite_count, kills) # Print diagnostics about the rewards distribution in the initial population
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
        sigma, sigma_decay, sigma_min, sigma_max, stagnation_limit, population_size,
        stagnation_ratio, generations, restart_ratio_min, restart_ratio_max, reference_std
    )
    new_best_found = False
    
    # Main evolutionary loop
    for generation in range(2, generations + 1):
        print(f"\n---------------------------------------------")
        print(f"\nGeneration {generation}/{generations}")

        # Select elites from current population (based on current rewards)
        elite_idx = np.argsort(rewards)[::-1][:elite_count]
        elites = [deepcopy(population[i]) for i in elite_idx]
        elite_rewards_carried = rewards[elite_idx].copy()  # carry forward without re-evaluation
        elite_kills_carried = kills[elite_idx].copy()  # carry forward without re-evaluation
        best_candidate = deepcopy(population[elite_idx[0]])
        best_candidate_reward = rewards[elite_idx[0]]

        # Build new population by reproduction (parents sampled w.r.t current rewards)
        new_population = []
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament(population, tournament_k, rewards)
            parent2 = tournament(population, tournament_k, rewards)

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
        offspring_rewards, offspring_kills = evaluate_population(agent_class, population[elite_count:])
        rewards = np.concatenate([elite_rewards_carried, offspring_rewards])
        kills = np.concatenate([elite_kills_carried, offspring_kills])
        debug_evaluate_population(rewards, tournament_k, elite_count, kills)

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
            sigma, sigma_decay, sigma_min, sigma_max, stagnation_limit, population_size,
            stagnation_ratio, generations, restart_ratio_min, restart_ratio_max, reference_std
        )
        new_best_found = False

    make_evolution_plot(best_rewards, mean_rewards,  True)
    inst = MLPAgent()
    inst.set_param_vector(best_params)
    return best_params

# Seeds to use: 42, 123, 999, 2024, 0
if __name__ == "__main__":
    np.random.seed(int(sys.argv[1]))
    torch.random.manual_seed(int(sys.argv[1]))
    genetic_algorithm()