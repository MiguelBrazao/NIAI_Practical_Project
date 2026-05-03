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

# Optional evaluation trace for reward composition diagnostics
_eval_debug_env = os.environ.get('EVAL_DEBUG', '0').lower()
EVAL_DEBUG = _eval_debug_env in ('1', 'true', 'yes')

port_list = [4242 + i for i in range(N_PROCESSES)]


def evaluate_agent(agent, task, episodes=1, max_fps=-1):
    """
    Evaluates the agent on the task for a given number of episodes.
    Returns (average_reward, total_kills).
    """
    exp = marioai.Experiment(task, agent)

    # Speed up simulation for training
    exp.max_fps = max_fps
    
    total_reward = 0
    total_kills = 0

    for _ in range(episodes):
        episode_reward = 0
        task.level_difficulty = 0
        task.kill_count = 0  # reset kills once per outer episode, not per sub-episode

        # Try up to 3 levels of increasing difficulty
        previous_kill_count = 0
        for sub_episode_idx in range(1, 4):
            prev_episode_reward = episode_reward
            kill_count_before_sub = int(getattr(task, 'kill_count', 0))
            exp.doEpisodes(1)

            # If Mario died, discard any kills counted during this sub-episode
            # (deaths-by-goomba trigger the same touch+disappear signal as stomps).
            if task.status != 1:
                task.kill_count = kill_count_before_sub

            # Apply kill multiplier to the base terminal reward NOW, after kills
            # have been corrected. base_terminal_reward was stored without kill
            # bonus so we can safely recompute it here.
            base = getattr(task, 'base_terminal_reward', 0.0)
            kill_bonus = base * (task.kill_count * task.kill_multiplier)
            task.cum_reward += kill_bonus

            episode_reward += task.cum_reward

            # Calculate kills for this sub-episode by looking at the change in kill_count
            sub_episode_kills = max(0, int(getattr(task, 'kill_count', 0)) - previous_kill_count)

            # Update previous_kill_count for the next iteration
            previous_kill_count = int(getattr(task, 'kill_count', 0))

            if EVAL_DEBUG:
                sub_reward = episode_reward - prev_episode_reward
                print(
                    f"[EVAL DEBUG] sub={sub_episode_idx} "
                    f"difficulty={task.level_difficulty} "
                    f"status={task.status} "
                    f"terminal_distance={getattr(task, 'last_terminal_distance', 0.0):.2f} "
                    f"kill_count={getattr(task, 'kill_count', 0)} "
                    f"sub_kills={sub_episode_kills} "
                    f"sub_reward={sub_reward:.2f} "
                    f"episode_reward_so_far={episode_reward:.2f}"
                )

            if task.status == 1: # WIN
                task.level_difficulty += 1
            else:
                break
        
        # Report cumulative kills for the whole outer episode.
        total_kills += int(getattr(task, 'kill_count', 0))

        total_reward += episode_reward
        
    return total_reward / episodes, total_kills


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
        reward, kills = evaluate_agent(worker_agent, worker_task)
    except Exception as e:
        print(f"Error in worker: {e}")
        reward = 0
        kills = 0
        
    return reward, kills


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

    rewards_list = [r for r, k in results]
    kills_list = [k for r, k in results]
    return np.array(rewards_list), np.array(kills_list, dtype=int)