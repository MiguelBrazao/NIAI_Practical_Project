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
    Runs through all 3 levels regardless of death, then reports
    per-level metrics, contiguous-survival totals, and grand totals.
    """
    import os

    # e.g. ga_move_forward_seed_42_1234.56.pkl
    #      random_search_hunter_seed_42_1234.56.pkl
    filename = os.path.basename(sys.argv[1])
    stem = filename.replace('.pkl', '')
    parts = stem.split('_seed_', 1)

    # Detect task from the prefix portion (robust to ga_ and random_search_ prefixes)
    task_key = 'move_forward' if 'move_forward' in parts[0] else 'hunter'

    # 42
    seed = int(parts[1].split('_')[0])

    TaskClass = MoveForwardTask if task_key == 'move_forward' else HunterTask

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = MLPAgent()
    task = TaskClass(visualization=True, port=4243, init_mario_mode=0)

    with open(f'{sys.argv[1]}', 'rb') as f:
        best_params = pkl.load(f)

    agent.set_param_vector(best_params)

    per_level, contiguous, grand = evaluation.evaluate_agent_detailed(agent, task, max_fps=60)

    print("\n=== Per-Level Results ===")
    for lv in per_level:
        status_str = "WIN " if not lv['died'] else "DIED"
        print(f"  Level {lv['level']}: {status_str}  |  Reward={lv['reward']:.3f}  |  Kills={lv['kills']}  |  Coins={lv['coins']}  |  Distance={lv['distance']:.1f}")

    deaths = [lv['level'] for lv in per_level if lv['died']]
    print(f"\nDied on level(s): {deaths if deaths else 'none'}")

    if contiguous['levels'] > 0:
        survived_range = f"0–{contiguous['levels'] - 1}"
    else:
        survived_range = "none"
    print(f"\n=== Contiguous Survival Total (levels {survived_range}) ===")
    print(f"  Levels: {contiguous['levels']}  |  Reward={contiguous['reward']:.3f}  |  Kills={contiguous['kills']}  |  Coins={contiguous['coins']}  |  Distance={contiguous['distance']:.1f}")

    print(f"\n=== Grand Total (all levels) ===")
    print(f"  Levels completed: {grand['levels_completed']}  |  Reward={grand['reward']:.3f}  |  Kills={grand['kills']}  |  Coins={grand['coins']}  |  Distance={grand['distance']:.1f}")


if __name__ == '__main__':
    evaluate_mlp_agent()
