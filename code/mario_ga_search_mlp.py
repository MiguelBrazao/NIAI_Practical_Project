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


def save_best_agent(params, reward):
    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
    inst = MLPAgent()
    inst.set_param_vector(params)                      # update mlp weights first
    with open(f'data/mlp_best_agents/ga_seed_{sys.argv[1]}_{reward:.3f}.pkl', 'wb') as f:
        pkl.dump(inst.get_param_vector(), f)               # save the actual agent param vector
    print(f"Saved new best agent with reward {reward:.3f}")


def debug_evaluate_population(rewards, population):
    # DIAGNOSTICS
    print("REWARDS summary: min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
        rewards.min(), rewards.max(), rewards.mean(), rewards.std()), flush=True)
    top5 = np.argsort(rewards)[::-1][:5]
    print("TOP5 indices:", top5, "TOP5 rewards:", rewards[top5], flush=True)
    print("TOP5 indices:", top5, "TOP5 rewards:", rewards[top5], flush=True)
    # Visual check of top candidate (single episode)
    top_idx = top5[0]
    inst = MLPAgent()
    inst.set_param_vector(population[top_idx])
    from evaluation import evaluate_agent, TASK_TO_SOLVE
    task = TASK_TO_SOLVE(visualization=True, port=4243)
    print("Running visual evaluation of best candidate...", flush=True)
    vis_reward = evaluate_agent(inst, task, episodes=1)
    print("VISUAL EVAL REWARD:", vis_reward, flush=True)


# Genetic Algorithm with Tournament Selection, Uniform Crossover, Gaussian Mutation, and Elitism.
# Population_size is the number of candidate solutions (MLP parameter vectors) in each generation: currently set to 20, which means we evaluate 20 different MLP parameter sets each generation.
# Generations is how many iterations of the evolutionary process to run: currently set to 10, which means we will evolve the population for 10 generations.
# Tournament_k is the number of individuals in each tournament: currently set to 2, which means we randomly select 2 individuals and pick the best among them as a parent.
# Crossover_rate is the probability of performing crossover: currently set to 0.8, which means that 80% of the time we will create a child by combining two parents, and 20% of the time we will just copy one parent (no crossover).
# Sigma is the mutation strength: currently set to 0.1, which means mutations will add noise with std dev of 0.1 to the parameters.
# Mutation_rate is the probability of mutating each parameter: currently set to 0.2, which means each parameter has a 20% chance of being mutated.
# Elite_count is how many top individuals to carry over unchanged: currently set to 1, which means the best individual from each generation will be directly copied to the next generation without any mutation or crossover.
# Crossover_mask_prob is the per-gene probability of taking a gene from parent1 during crossover: currently set to 0.5, which means that for each parameter, there is a 50% chance it will come from parent1 and a 50% chance it will come from parent2 (uniform crossover).
def genetic_algorithm(population_size=200, generations=300, tournament_k=4,
                        crossover_rate=0.9, sigma=0.15, mutation_rate=0.2,
                        elite_count=4, crossover_mask_prob=0.6):
    """
    Evolve MLP weights using a Genetic Algorithm with:
        - Tournament selection (parent selection)
        - Uniform crossover
        - Gaussian mutation
        - Elitism (survivor selection)

    Hyperparameter examples:
        - Quick debug (fast, single-process, low cost):
            genetic_algorithm(population_size=20, generations=10, tournament_k=2,
                            crossover_rate=0.8, sigma=0.1, mutation_rate=0.2,
                            elite_count=1, crossover_mask_prob=0.5)

        - Regular debug (fast, single-process, more stable):
            genetic_algorithm(population_size=40, generations=100,
                            tournament_k=3, crossover_rate=0.8,
                            sigma=0.10, mutation_rate=0.20,
                            elite_count=4, crossover_mask_prob=0.5)

        - Baseline (reasonable tradeoff):
            genetic_algorithm(population_size=80, generations=100, tournament_k=3,
                            crossover_rate=0.8, sigma=0.25, mutation_rate=0.3,
                            elite_count=2, crossover_mask_prob=0.5)

        - Thorough (more compute, slower convergence but better search):
            genetic_algorithm(population_size=200, generations=300, tournament_k=4,
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
    
    # Initialize population with random parameter vectors
    num_params = len(MLPAgent().get_param_vector())  # get number of parameters from a fresh agent instance
    population = [np.random.randn(num_params) for _ in range(population_size)] # initialize population with random parameter vectors
    print(f"\n--- Generation 1/{generations} ---") # Evaluate initial population
    rewards = evaluate_population(agent_class, population) # Evaluate all individuals in the population and get their rewards
    debug_evaluate_population(rewards, population) # Print diagnostics about the rewards distribution in the initial population
    best_params = deepcopy(population[np.argmax(rewards)]) # Keep track of the best parameters found so far
    best_reward = rewards.max() # Keep track of the best reward found so far
    save_best_agent(best_params, best_reward) # Save the best agent from the initial population
    print(f"Generation 1: Best = {rewards.max():.3f}  Mean = {rewards.mean():.3f}") # Log the best and mean reward of the initial population
    best_rewards = [best_reward] # Track the best reward of each generation for plotting
    mean_rewards = [rewards.mean()] # Track the mean reward of each generation for plotting
    new_best_found = False # Flag to track if a new best agent was found in the current generation (used for saving)

    # Main evolutionary loop
    for generation in range(2, generations + 1):
        print(f"\n--- Generation {generation}/{generations} ---")

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
            child[mask] += sigma * np.random.randn(mask.sum())

            new_population.append(child)

        # Apply elitism now (copy elites into new population)
        # Replace the first positions with elites (keeps population size)
        for i, e in enumerate(elites):
            new_population[i] = deepcopy(e)

        # Advance to new population and evaluate it
        population = new_population
        rewards = evaluate_population(agent_class, population)
        debug_evaluate_population(rewards, population) # Print diagnostics about the rewards distribution in the new population

        # Track global best AFTER offspring generation (use saved best_candidate from previous population)
        if best_candidate_reward > best_reward:
            best_reward = best_candidate_reward
            best_params = best_candidate
            new_best_found = True

        # Save the best params only AFTER weights have been applied to an agent (and after offspring generation)
        if new_best_found:
            save_best_agent(best_params, best_reward)
            new_best_found = False

        print(f"Generation {generation}: Best = {rewards.max():.3f}  Mean = {rewards.mean():.3f}")
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