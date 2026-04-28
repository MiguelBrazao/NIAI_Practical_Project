import numpy as np
import torch
import marioai
from agents import MLPAgent, CodeAgent
from tasks import MoveForwardTask, HunterTask
import pickle as pkl
import sys
import inspect
import data.gp_best_agents.mario_best as mario_best

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
    agent = MLPAgent()
    task = HunterTask(visualization=True, port=4243)
    exp = marioai.Experiment(task, agent)


    with open(f'{sys.argv[1]}', 'rb') as f:
        best_params = pkl.load(f)


    agent.set_param_vector(best_params)
    exp.max_fps = 60
    task.env.level_type = 0
    rewards = exp.doEpisodes()
    print(sum(rewards[0]))


if __name__ == '__main__':
    evaluate_mlp_agent()
    #evaluate_code_agent()