import numpy as np
import torch
import marioai
from agents import MLPAgent, CodeAgent
from tasks import MoveForwardTask, HunterTask
import pickle as pkl
import sys
import inspect
import argparse
from pathlib import Path
import data.gp_best_agents.mario_best as mario_best
import re

def evaluate_code_agent():
    action = inspect.getsource(mario_best.corre)
    agent = CodeAgent()
    agent.action_function = "def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  action[Mario.KEY_RIGHT] = 0\n  if landscape[11 + -1][11 + 2] != 21:\n    action[Mario.KEY_JUMP] = 1\n    if can_jump:\n      action[Mario.KEY_DOWN] = 0\n    else:\n      if on_ground:\n        if can_jump:\n          action[Mario.KEY_RIGHT] = 1\n        else:\n          action[Mario.KEY_JUMP] = 0\n        action[Mario.KEY_DOWN] = 1\n    action[Mario.KEY_RIGHT] = 1\n  else:\n    if on_ground:\n      action[Mario.KEY_LEFT] = 1\n    else:\n      if enemies[11 + -2][11 + -2] == Sprite.KIND_GREEN_KOOPA_WINGED:\n        action[Mario.KEY_RIGHT] = 0\n      else:\n        action[Mario.KEY_DOWN] = 1\n      action[Mario.KEY_RIGHT] = 0" #"def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  action[Mario.KEY_RIGHT] = 1\n  if enemies[11 + -1][11 + 0] != Sprite.KIND_RED_KOOPA_WINGED:\n    action[Mario.KEY_JUMP] = 1\n    if can_jump:\n      if enemies[11 + -1][11 + 0] != Sprite.KIND_RED_KOOPA_WINGED:\n        action[Mario.KEY_DOWN] = 1\n    else:\n      if on_ground:\n        action[Mario.KEY_JUMP] = 0\n    action[Mario.KEY_DOWN] = 0\n    if can_jump:\n      action[Mario.KEY_SPEED] = 1" # "def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  if can_jump:\n    action[Mario.KEY_JUMP] = 1\n    if enemies[11 + 1][11 + 1] != Sprite.KIND_GOOMBA_WINGED:\n      if on_ground:\n        action[Mario.KEY_SPEED] = 1\n    else:\n      action[Mario.KEY_JUMP] = 0\n  else:\n    if on_ground:\n      action[Mario.KEY_DOWN] = 0\n    else:\n      action[Mario.KEY_DOWN] = 1\n      action[Mario.KEY_SPEED] = 0\n      if can_jump:\n        if enemies[11 + 0][11 + -1] == Sprite.KIND_BULLET_BILL:\n          action[Mario.KEY_RIGHT] = 0\n          action[Mario.KEY_RIGHT] = 1\n      else:\n        action[Mario.KEY_JUMP] = 1\n    if on_ground:\n      if on_ground:\n        if can_jump:\n          action[Mario.KEY_JUMP] = 1\n        else:\n          action[Mario.KEY_LEFT] = 1\n        action[Mario.KEY_DOWN] = 0\n      else:\n        action[Mario.KEY_DOWN] = 1\n      action[Mario.KEY_SPEED] = 1\n    else:\n      action[Mario.KEY_RIGHT] = 1\n    action[Mario.KEY_SPEED] = 0\n  action[Mario.KEY_RIGHT] = 1" #"def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  if can_jump:\n    action[Mario.KEY_JUMP] = 1\n    if on_ground:\n      if landscape[11 + -1][11 + -1] != 20:\n        action[Mario.KEY_RIGHT] = 0\n      else:\n        action[Mario.KEY_JUMP] = 1\n    else:\n      action[Mario.KEY_LEFT] = 1\n      action[Mario.KEY_JUMP] = 1\n  else:\n    action[Mario.KEY_RIGHT] = 0\n    if can_jump:\n      if can_jump:\n        if enemies[11 + -1][11 + 0] == Sprite.KIND_GREEN_KOOPA_WINGED:\n          if on_ground:\n            if on_ground:\n              action[Mario.KEY_DOWN] = 1\n            else:\n              action[Mario.KEY_RIGHT] = 1\n        else:\n          action[Mario.KEY_SPEED] = 0\n          action[Mario.KEY_DOWN] = 0\n    else:\n      action[Mario.KEY_RIGHT] = 1\n  action[Mario.KEY_SPEED] = 1\n  if landscape[11 + 1][11 + -1] != 21:\n    action[Mario.KEY_DOWN] = 0\n  action[Mario.KEY_SPEED] = 1" #"def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  if on_ground:\n    if can_jump:\n      action[Mario.KEY_JUMP] = 1\n      if enemies[11 + 1][11 + 1] != Sprite.KIND_SPIKY:\n        action[Mario.KEY_RIGHT] = 0\n      else:\n        action[Mario.KEY_JUMP] = 1\n    else:\n      action[Mario.KEY_SPEED] = 1\n      action[Mario.KEY_LEFT] = 1\n    if landscape[11 + 0][11 + -1] != 0:\n      action[Mario.KEY_DOWN] = 0\n      action[Mario.KEY_DOWN] = 0\n    else:\n      if on_ground:\n        action[Mario.KEY_DOWN] = 0\n      action[Mario.KEY_RIGHT] = 1\n  else:\n    action[Mario.KEY_RIGHT] = 1\n    action[Mario.KEY_JUMP] = 1" #def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  if on_ground:\n    if can_jump:\n      action[Mario.KEY_JUMP] = 1\n  else:\n    if landscape[11 + 1][11 + 1] != 20:\n      action[Mario.KEY_LEFT] = 1\n      action[Mario.KEY_LEFT] = 1\n      action[Mario.KEY_LEFT] = 0\n    else:\n      action[Mario.KEY_RIGHT] = 0\n      action[Mario.KEY_DOWN] = 1\n    action[Mario.KEY_JUMP] = 1\n    action[Mario.KEY_SPEED] = 0\n    if can_jump:\n      action[Mario.KEY_RIGHT] = 0\n      action[Mario.KEY_RIGHT] = 0\n    else:\n      action[Mario.KEY_JUMP] = 1\n      action[Mario.KEY_LEFT] = 0\n      if landscape[11 + 1][11 + 0] != 21:\n        action[Mario.KEY_RIGHT] = 1\n      else:\n        action[Mario.KEY_JUMP] = 1\n        action[Mario.KEY_LEFT] = 1\n  action[Mario.KEY_SPEED] = 1\n  if can_jump:\n    if landscape[11 + 1][11 + 0] != -10:\n      action[Mario.KEY_JUMP] = 1\n    else:\n      action[Mario.KEY_SPEED] = 1" #"def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  action[Mario.KEY_RIGHT] = 1\n  if can_jump:\n    action[Mario.KEY_JUMP] = 1\n  else:\n    action[Mario.KEY_SPEED] = 1\n  action[Mario.KEY_LEFT] = 0\n  if on_ground:\n    action[Mario.KEY_LEFT] = 1\n  else:\n    action[Mario.KEY_JUMP] = 1" #"def corre(Mario, Sprite, landscape, enemies, can_jump, on_ground, action):\n  if can_jump:\n    if on_ground:\n      action[Mario.KEY_JUMP] = 1\n      action[Mario.KEY_SPEED] = 0\n      action[Mario.KEY_SPEED] = 0\n    else:\n      action[Mario.KEY_DOWN] = 1\n  else:\n    action[Mario.KEY_SPEED] = 0\n  if on_ground:\n    if can_jump:\n      if can_jump:\n        action[Mario.KEY_DOWN] = 0\n      else:\n        if can_jump:\n          action[Mario.KEY_JUMP] = 1\n        else:\n          action[Mario.KEY_SPEED] = 0\n    if on_ground:\n      action[Mario.KEY_LEFT] = 1\n  else:\n    action[Mario.KEY_RIGHT] = 1\n    if enemies[11 + 0][11 + 0] != Sprite.KIND_GREEN_KOOPA_WINGED:\n      action[Mario.KEY_JUMP] = 1\n    else:\n      if can_jump:\n        action[Mario.KEY_DOWN] = 0\n      else:\n        action[Mario.KEY_RIGHT] = 1\n        if on_ground:\n          action[Mario.KEY_LEFT] = 1"
    #agent.action_function = action
    task = HunterTask(visualization=True, port=4243, init_mario_mode=0, level_difficulty=0)
    exp = marioai.Experiment(task, agent)
    exp.max_fps = 60
    task.env.level_type = 0
    rewards = sum(exp.doEpisodes()[0])

    task.level_difficulty += 1
    print(task.level_difficulty)
    exp.max_fps = 60
    task.env.level_type = 0
    rewards += sum(exp.doEpisodes()[0])

    task.level_difficulty += 1
    print(task.level_difficulty)
    exp.max_fps = 60
    task.env.level_type = 0
    rewards += sum(exp.doEpisodes()[0])
        
    print(rewards)


def evaluate_mlp_agent():
    # kept for API compatibility; prefer calling via CLI / main
    return evaluate_mlp_agent_with_params()


def evaluate_mlp_agent_with_params(agent_path: str = None, task_name: str = 'move_forward', visualize: bool = True, port: int = 4243):
    agent = MLPAgent()
    TaskClass = MoveForwardTask if task_name == 'move_forward' else HunterTask
    task = TaskClass(visualization=visualize, port=port)
    exp = marioai.Experiment(task, agent)

    # Choose pickle: CLI arg preferred; otherwise pick file with highest reward encoded in filename
    if agent_path is None:
        candidates = list(Path('data/mlp_best_agents').glob('*.pkl'))
        if not candidates:
            print('No agent file provided and none found in data/mlp_best_agents/. Provide --agent <path>.')
            return

        # Filenames expected like: <model>_seed_<seed>_<float reward>.pkl
        best = None
        best_reward = -float('inf')
        pattern = re.compile(r'(-?\d+(?:\.\d+)?)$')  # match final signed float
        for c in candidates:
            m = pattern.search(c.stem)
            if not m:
                continue
            try:
                r = float(m.group(1))
            except Exception:
                continue
            if r > best_reward:
                best_reward = r
                best = c

        if best is None:
            # fallback to newest if no parsable reward found
            pick = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f'No parsable reward in filenames; using newest agent: {pick}')
            p = pick
        else:
            print(f'No path provided; using agent with highest reward {best_reward}: {best}')
            p = best
    else:
        p = Path(agent_path)

    with open(p, 'rb') as f:
        best_params = pkl.load(f)

    agent.set_param_vector(best_params)
    exp.max_fps = 60
    task.env.level_type = 0
    rewards = exp.doEpisodes()
    print(sum(rewards[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a saved MLP agent.')
    parser.add_argument('--agent', '-a', default=None, help='Path to agent pickle file. If omitted, uses newest in data/mlp_best_agents.')
    parser.add_argument('--task', '-t', choices=['move_forward', 'hunter'], default='move_forward', help='Which task to evaluate (default: move_forward).')
    parser.add_argument('--port', type=int, default=4243, help='Port to connect to Mario server (default: 4243).')
    parser.add_argument('--no-visual', dest='visualize', action='store_false', help='Disable visualization.')
    args = parser.parse_args()

    evaluate_mlp_agent_with_params(agent_path=args.agent, task_name=args.task, visualize=args.visualize, port=args.port)
    # evaluate_code_agent()