import torch
import torch.nn as nn
import numpy as np
import marioai



class MoveForwardTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MoveForward"


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

        # last_obs is None on the very first step
        if last_obs is None or current_obs.mario_pos is None or last_obs.mario_pos is None:
            return 0

        reward = 0

        cur_x, cur_y = current_obs.mario_pos
        last_x, last_y = last_obs.mario_pos
        distance = current_obs.distance

        if cur_x > last_x:
            reward += 1         # Reward for moving forward
        else:
            reward -= 5         # Penalty for not moving or moving backward

        if cur_x >= 3000:       # Bonus for reaching the end of the level
            reward += 100

        if cur_y < 0:           # Penalty for falling under a certain threshold (e.g., falling into a pit)
            reward -= 50

        if cur_y > last_y:      # Reward for jumping
            reward += 20

        reward += distance      # Reward for overall distance progress

        return reward