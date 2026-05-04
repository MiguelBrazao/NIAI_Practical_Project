import numpy as np
import torch
import marioai
from agents import MLPAgent
from tasks import MoveForwardTask, HunterTask
import pickle as pkl
import sys
import inspect
import data.gp_best_agents.mario_best as mario_best
import evaluation 


def evaluate_mlp_agent():
    """
    Evaluates the best MLP agent saved in a pickle file. 
    The file name is expected to contain the task type and 
    random seed used during training, which are extracted 
    to configure the evaluation environment. The agent's 
    parameters are loaded from the file, and the agent 
    is evaluated on the specified task for a single 
    episode with visualization enabled. The final 
    reward and kill count are printed at the 
    end of the evaluation.
    """
    import os

    # e.g. ga_move_forward_seed_42_1234.56.pkl
    filename = os.path.basename(sys.argv[1])           

    # ga_move_forward_seed_42_1234.56
    stem = filename.replace('.pkl', '')

    # ['ga_move_forward', '42_1234.56']                
    parts = stem.split('_seed_', 1)                    

    # 'move_forward'
    task_key = parts[0].removeprefix('ga_')          

    # 42  
    seed = int(parts[1].split('_')[0])                 

    TaskClass = MoveForwardTask if task_key == 'move_forward' else HunterTask

    # Seed was the numpy/torch random seed 
    # used during GA training, not a level seed.
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = MLPAgent()
    task = TaskClass(visualization=True, port=4243, init_mario_mode=0)

    with open(f'{sys.argv[1]}', 'rb') as f:
        best_params = pkl.load(f)

    agent.set_param_vector(best_params)

    reward, kills, coins, distance, levels = evaluation.evaluate_agent(agent, task, episodes=1, max_fps=60)
    print(f"Reward: {reward:.3f}  |  Kills: {kills}  |  Coins: {coins}  |  Distance: {distance:.1f}  |  Levels completed: {levels}")


if __name__ == '__main__':
    evaluate_mlp_agent()
