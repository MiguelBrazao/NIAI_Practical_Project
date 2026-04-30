import torch
import torch.nn as nn
import numpy as np
import marioai
import tasks.rewards as rewards

class MoveForwardTask(marioai.Task, rewards.Rewards):
    def __init__(self, *args, **kwargs):
        marioai.Task.__init__(self, *args, **kwargs)
        rewards.Rewards.__init__(self)
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
        self.distance()                             # Compute final raw reward based on distance traveled forward (primary objective)
        return self.reward                          # Return the computed reward and reset internal state for next step 


    def reset(self):
        marioai.Task.reset(self)
        rewards.Rewards.reset(self)


    def perform_action(self, action):
        marioai.Task.perform_action(self, action)
        rewards.Rewards.perform_action(self, action)


    def get_sensors(self):
        return rewards.Rewards.get_sensors(self)