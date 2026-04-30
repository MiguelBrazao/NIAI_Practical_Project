import torch
import torch.nn as nn
import numpy as np
import marioai
import tasks.rewards as rewards


class HunterTask(marioai.Task, rewards.Rewards):
    def __init__(self, *args, **kwargs):
        marioai.Task.__init__(self, *args, **kwargs)
        rewards.Rewards.__init__(self)
        self.name = "Hunter"

    def compute_reward(self, current_obs, last_obs):
        """
        Computes the reward for the current state of the game based on Mario's actions 
        and the environment changes between the current and last observations.
        This function evaluates Mario's progress, interactions with enemies, and overall 
        performance to calculate a reward value. The reward is used as the fitness function for the evolutionary algorithm.
        Parameters:
        - current_obs: The current observation of the game state;
        - last_obs: The previous observation of the game state;
        Returns:
        - reward (float): The computed reward value based on the game state changes.
        Notes for Students:
        - This function is critical for defining the algorithm behavior. The reward function 
          directly impacts the fitness evaluation of the AI.
        - You are encouraged to edit and experiment with this function to design a reward 
          system that aligns with the objectives of the project.
        - Consider the balance between encouraging progress, rewarding kills, and penalizing 
          undesirable behaviors (e.g., cowardice or reckless actions).
        """

        self.progression_reward_ratio = 1/1200      # test between 2400 and 600 -- accumulates after episode of steps
        self.kills_reward_value = 5.0               # test between 1.0 and 10.0 -- accumulates per step

        self.observations(current_obs, last_obs)    # Update internal state with current and last observations for reward calculations and get_sensors access
        self.progression()                          # Compute reward based on level progression (distance traveled forward) -- primary objective to encourage forward movement and level completion
        self.kills()                                # Reward for collecting power-ups (secondary: encourages exploration to find enemies)


    def reset(self):
        marioai.Task.reset(self)
        rewards.Rewards.reset(self)
        

    def perform_action(self, action):
        marioai.Task.perform_action(self, action)
        rewards.Rewards.perform_action(self, action)
        

    def get_sensors(self):
        return rewards.Rewards.get_sensors(self)




























        # if last_obs is None or current_obs.mario_pos is None or last_obs.mario_pos is None:
        #     return 0

        # reward = 0

        # cur_x = current_obs.mario_pos[0]
        # last_x = last_obs.mario_pos[0]

        # # Primary objective: reward enemy kills (fewer enemies in scene = kill happened)
        # ENEMY_VALUES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}
        # cur_enemies  = sum(1 for v in current_obs.level_scene.flatten() if v in ENEMY_VALUES)
        # last_enemies = sum(1 for v in last_obs.level_scene.flatten()    if v in ENEMY_VALUES)
        # kills = last_enemies - cur_enemies
        # if kills > 0:
        #     reward += kills * 5    # Strong reward per enemy killed

        # # Secondary: small reward for moving forward (to avoid static camping)
        # if cur_x > last_x:
        #     reward += 0.5

        # return reward