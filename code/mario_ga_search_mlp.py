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


@contextmanager
def timer_context(label):
    start = time.perf_counter()
    try:
        # Yields control back to the code inside the 'with' block
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}] Elapsed time: {end - start:.4f} seconds")


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


def make_evolution_plot(best, mean, save=False):
    """
    Constructs a plot of the best and mean rewards over generations. 
    The plot is updated in real-time during the evolution process.
    """
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
    """ 
    Performs tournament selection on the population.

    Args:
        population (list): The current population of individuals.
        tournament_k (int): The number of individuals to participate in each tournament.
        rewards (np.ndarray): The rewards associated with each individual.
        replace (bool): Whether to allow replacement in the tournament selection.

    Returns:
        The best individual from the tournament.
    """
    # only consider individuals with valid rewards for tournament 
    # selection (exclude -inf from reinitialized individuals)
    eligible = np.where(np.isfinite(rewards))[0] 

    # if not enough eligible individuals, fallback
    # to entire population to avoid errors
    pool = eligible if len(eligible) >= 2 else np.arange(len(population)) 

    # select tournament_k individuals from the eligible pool without 
    # replacement (or with replacement if not enough eligible)
    idx = np.random.choice(pool, min(tournament_k, len(pool)), replace=replace)

    # return a copy of the best individual from the tournament 
    return deepcopy(population[idx[np.argmax(rewards[idx])]]) 


def save_best_agent(params, reward):
    """
    Saves the best agent's parameters to a file.

    Args:
        params (np.ndarray): The parameter vector of the agent.
        reward (float): The reward achieved by the agent.
    """

    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
    
    # update mlp weights first
    inst = MLPAgent()
    inst.set_param_vector(params) 

    mode = 'hunter' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'move_forward'
    with open(f'data/mlp_best_agents/ga_{mode}_seed_{sys.argv[1]}_{reward:.3f}.pkl', 'wb') as f:
        # save the actual agent param vector
        pkl.dump(inst.get_param_vector(), f)

    print(f"\nSaved new best agent with reward {reward:.3f}")


def debug_evaluate_population(rewards, tournament_k, elite_count, kills=None):
    """
    Prints debug information about the population's 
    rewards, including statistics and top performers.
    """
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
        best_kills = kills[np.argmax(rewards)]
        if best_kills > 0:
            print(f"\nBest Candidate Kills: {best_kills}")


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

    print("\n=============================================")

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
        print(f"\n=============================================")

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
    # keep class for
    # evaluate_population 
    # (used by workers)
    agent_class = MLPAgent  

    task_label = 'HUNTER' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'MOVE FORWARD'

    # convert ratio to absolute generation count
    stagnation_limit = max(1, round(stagnation_ratio * generations))  

    print(f"\n=============================================")
    print(f"\n{task_label} GA Search with MLP Agent")
    print(f"\n=============================================\n")
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
    print(f"\n=============================================")
    print(f"\nStagnation Limit is {stagnation_limit} generations")
    print(f"\n=============================================")
    print(f"\nGeneration 1/{generations}")

    # Timing bookkeeping
    generation_times = []
    total_start = time.perf_counter()

    # Initialize population with random parameter vectors    
    # get number of parameters from a fresh agent instance
    num_params = len(MLPAgent().get_param_vector())  

    # initialize population with random parameter vectors
    population = [np.random.randn(num_params) for _ in range(population_size)]

    # Evaluate all individuals in the population and get their rewards
    gen_start = time.perf_counter()
    rewards, kills = evaluate_population(agent_class, population)

    # Keep track of the best parameters found so far 
    best_params = deepcopy(population[np.argmax(rewards)]) 

    # Keep track of the best 
    # reward found so far
    best_reward = rewards.max() 

    # baseline std from fully random 
    # generation 1, used to normalise 
    # adaptive restart ratio
    reference_std = rewards.std()  
    
    # Print diagnostics about the rewards distribution in the initial population
    debug_evaluate_population(rewards, tournament_k, elite_count, kills) 

    # Save the best agent from the initial population
    save_best_agent(best_params, best_reward) 

    # Flag to track if a new best agent was found 
    # in the current generation (used for saving)
    new_best_found = True 

    # Track the best reward of 
    # each generation for plotting
    best_rewards = [best_reward] 

    # Track the mean reward of 
    # each generation for plotting
    mean_rewards = [rewards.mean()] 

    # generations since 
    # last improvement
    stagnation_count = 0   

    # sigma decays each 
    # generation toward 
    # sigma_min
    current_sigma = sigma  
    
    # Sigma decay + stagnation restart
    current_sigma, stagnation_count, population, rewards = update_sigma_stagnation(
        new_best_found, current_sigma, stagnation_count,
        population, rewards, num_params,
        sigma, sigma_decay, sigma_min, sigma_max, stagnation_limit, population_size,
        stagnation_ratio, generations, restart_ratio_min, restart_ratio_max, reference_std
    )
    gen_elapsed = time.perf_counter() - gen_start
    generation_times.append(gen_elapsed)
    print(f"\n=============================================")
    print(f"\nGeneration {1} time: {gen_elapsed:.3f}s")
    print(f"Total so far: {sum(generation_times):.3f}s")
    print(f"Mean/gen: {np.mean(generation_times):.3f}s")
    print(f"Std/gen: {np.std(generation_times):.3f}s")
    new_best_found = False
    
    # Main evolutionary loop
    for generation in range(2, generations + 1):
        gen_start = time.perf_counter()
        print(f"\n=============================================")
        print(f"\nGeneration {generation}/{generations}")

        # Select elites from current population (based on current rewards)
        elite_idx = np.argsort(rewards)[::-1][:elite_count]
        elites = [deepcopy(population[i]) for i in elite_idx]
        elite_rewards_carried = rewards[elite_idx].copy() 
        elite_kills_carried = kills[elite_idx].copy() 
        best_candidate = deepcopy(population[elite_idx[0]])
        best_candidate_reward = rewards[elite_idx[0]]

        # Build new population by reproduction 
        # (parents sampled w.r.t current rewards)
        new_population = []
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament(population, tournament_k, rewards)
            parent2 = tournament(population, tournament_k, rewards)

            # Uniform Crossover
            if np.random.rand() < crossover_rate:
                # per-gene probability to take gene 
                # from parent1 (default 0.5 = uniform crossover)
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

        # Advance to new population; evaluate only 
        # offspring, carry elite rewards forward
        population = new_population
        offspring_rewards, offspring_kills = evaluate_population(agent_class, population[elite_count:])
        rewards = np.concatenate([elite_rewards_carried, offspring_rewards])
        kills = np.concatenate([elite_kills_carried, offspring_kills])
        debug_evaluate_population(rewards, tournament_k, elite_count, kills)

        # Track global best: check both the old elite 
        # (carried forward) and the new population's best
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

        # Save the best params only AFTER weights have been 
        # applied to an agent (and after offspring generation)
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
        gen_elapsed = time.perf_counter() - gen_start
        generation_times.append(gen_elapsed)
        
        print(f"\n=============================================")
        print(f"\nGeneration {generation} time: {gen_elapsed:.3f}s")
        print(f"Total so far: {sum(generation_times):.3f}s")
        print(f"Mean/gen: {np.mean(generation_times):.3f}s")
        print(f"Std/gen: {np.std(generation_times):.3f}s")
        new_best_found = False

    total_elapsed = time.perf_counter() - total_start
    print(f"\n=============================================")
    print(f"\nGA Timing Summary")
    print(f"Generations timed: {len(generation_times)}")
    print(f"Mean time per generation: {np.mean(generation_times):.3f}s")
    print(f"Std time per generation: {np.std(generation_times):.3f}s")
    print(f"Total time (all generations): {total_elapsed:.3f}s")
    print(f"\n=============================================\n")

    make_evolution_plot(best_rewards, mean_rewards,  True)
    inst = MLPAgent()
    inst.set_param_vector(best_params)
    return best_params

# Seeds to use: 42, 123, 999, 2024, 0
if __name__ == "__main__":
    seed = int(sys.argv[1])
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    method = 'hunter' if TASK_TO_SOLVE.__name__ == 'HunterTask' else 'move_forward'
    log_dir = Path("data")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ga_{method}_seed_{seed}.txt"

    with open(log_path, 'a', encoding='utf-8') as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            genetic_algorithm()
