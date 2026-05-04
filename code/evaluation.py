import marioai
from multiprocessing import Pool, Manager, current_process
from itertools import cycle
from agents import MLPAgent, CodeAgent
from tasks import MoveForwardTask, HunterTask
import numpy as np
import os

# Variable that configures 
# the number of parallel processes
N_PROCESSES = 5

# Task selection via environment variable: 'move_forward' or 'hunter'
_task_env = os.environ.get('TASK', 'hunter').lower()
TASK_TO_SOLVE = MoveForwardTask if _task_env == 'move_forward' else HunterTask

# Visualization toggle: set GRAPHICS=0 to disable visualization during training (default ON)
_vis_env = os.environ.get('GRAPHICS', '1').lower()
VISUALIZE = False if _vis_env in ('0', 'false', 'no') else True

port_list = [4242 + i for i in range(N_PROCESSES)]


def evaluate_agent(agent, task, episodes=1, max_fps=-1):
    """
    Evaluates the agent on the task for a given number of episodes.
    Returns (average_reward, total_kills, total_coins, total_distance, total_levels_completed).
    """
    exp = marioai.Experiment(task, agent)

    # Speed up simulation for training
    exp.max_fps = max_fps
    
    # We will accumulate rewards and kills 
    # across all episodes and return the averages.
    total_reward = 0
    total_kills = 0
    total_coins = 0
    total_distance = 0.0
    total_levels_completed = 0

    for _ in range(episodes):
        episode_reward = 0
        task.level_difficulty = 0
        
        # reset kills once per outer 
        # episode, not per sub-episode
        task.kill_count = 0  

        # Try up to 3 levels 
        # of increasing difficulty
        for _ in range(3):
            exp.doEpisodes(1)

            # Convert touch events from 
            # this sub-episode into kills.
            # Rule requested:
            # - if Mario does not die (WIN), count all touches
            # - if Mario dies, discard only the last touch (death hit)
            task.kill_count += int(getattr(task, 'sub_episode_touch_events', 0))

            # print(f"Confirmed kills this sub-episode: {getattr(task, 'sub_episode_touch_events', 0)}")

            # The base distance reward is already in task.cum_reward (added by
            # get_sensors at episode end). Here we only add the kill bonus on top,
            # and only when kill_rewards is enabled. When disabled, kills are still
            # tracked in kill_count for reporting but do not affect the reward.
            if getattr(task, 'kill_rewards', False):
                kill_bonus = getattr(task, 'base_terminal_reward', 0.0) * task.kill_count * task.kill_multiplier
                task.cum_reward += kill_bonus

            episode_reward += task.cum_reward

            # Accumulate distance and coins per sub-episode (both reset each level).
            mario_pos = getattr(task, 'last_mario_pos', None)
            total_distance += float(mario_pos[0]) if mario_pos is not None else 0.0
            total_coins += int(getattr(task, 'last_coins', 0))

            if task.status == 1: # WIN
                total_levels_completed += 1
                task.level_difficulty += 1
            else:
                break
        
        # Report cumulative stats for the whole outer episode.
        total_reward += episode_reward
        total_kills += int(getattr(task, 'kill_count', 0))
        
        
    return total_reward / episodes, total_kills, total_coins, total_distance, total_levels_completed


# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
# These exist independently inside EACH worker process.
worker_task = None 
worker_agent = None

def init_worker(agent_class):
    """
    This runs ONCE when each worker process starts.
    """
    global worker_agent, worker_task
    
    # Each worker needs to pick a port. Since we have 10 workers 
    # and 10 ports, we can use a trick to assign them.
    import multiprocessing
    # Get the index of the current worker (0 through 9)
    # Note: This is a hacky way to get a unique index; 
    # alternatively, use a shared Counter/Queue.
    worker_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    port = port_list[worker_idx % len(port_list)]
    
    #print(f"Worker initialized: Connecting once to port {port}...")

    worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=VISUALIZE, port=port, init_mario_mode=0)


def evaluate_individual(ind_info):
    """
    This runs for every individual in the population.
    It uses the GLOBALLY cached worker_task.
    """
    global worker_task, worker_agent
    
    # 1. Update the persistent agent with the new DNA
    if isinstance(worker_agent, MLPAgent):
        worker_agent.set_param_vector(ind_info)
    elif isinstance(worker_agent, CodeAgent):
        worker_agent.action_function = ind_info

    
    # 2. Run evaluation using the EXISTING connection
    # No "with", no "connect", just use the persistent object.
    try:
        reward, kills, coins, distance, levels = evaluate_agent(worker_agent, worker_task)
    except Exception as e:
        print(f"Error in worker: {e}")
        reward = 0
        kills = 0
        coins = 0
        distance = 0.0
        levels = 0
        
    return reward, kills, coins, distance, levels


def evaluate(agent_class, ind_info):
    global worker_agent, worker_task
    if worker_agent is None:
        worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=VISUALIZE, port=port_list[0])
    return evaluate_individual(ind_info)


def evaluate_population(agent, population):
    
    # Match processes to tasks to avoid one worker being idle or double-booking
    n_processes = N_PROCESSES

    # We pass 'tasks' to the initializer, so every worker picks one at startup
    with Pool(processes=n_processes, initializer=init_worker, initargs=(agent,)) as pool:
        # We only map the POPULATION. The tasks are already fixed in the workers.
        results = pool.map(evaluate_individual, population)
    
    worker_task = None

    rewards_list = [r for r, k, c, d, l in results]
    kills_list = [k for r, k, c, d, l in results]
    coins_list = [c for r, k, c, d, l in results]
    distance_list = [d for r, k, c, d, l in results]
    levels_list = [l for r, k, c, d, l in results]
    return np.array(rewards_list), np.array(kills_list, dtype=int), np.array(coins_list, dtype=int), np.array(distance_list), np.array(levels_list, dtype=int)