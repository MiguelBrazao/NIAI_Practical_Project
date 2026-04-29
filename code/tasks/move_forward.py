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
        self.environment(current_obs, last_obs)   # Update internal state based on observations (e.g., position, velocity, enemies)
        self.controls()                           # Reward for using the controls effectively (e.g., not pressing too many buttons at once or none at all), which encourages more strategic and less erratic actions.
        self.forward()                            # Reward for moving forward (primary objective) and penalize for moving backward or staying still (encourage progress towards the goal)
        self.jump(to_collect=False)               # Reward for jumping when he should (e.g., to avoid enemies or gaps) and penalize for unnecessary jumps (e.g., jumping in place or when on the ground without obstacles)
        self.duck(allow_ducking=False)            # Penalty for ducking when it's not allowed (e.g., when it doesn't help avoid a threat or is unnecessary), which encourages the agent to avoid unnecessary ducking that could lead to negative consequences. In this task, we can set allow_ducking=False to discourage ducking since it's not relevant for moving forward and could lead to more erratic behavior.
        self.obstacles()                          # Reward for successfully navigating obstacles (e.g., jumping over gaps or enemies) and penalize for failing to navigate them (e.g., falling into gaps or colliding with enemies), which encourages the agent to learn how to effectively deal with obstacles while moving forward.
        return self.reward                        # Return the computed reward and reset internal state for next step 


    def reset(self):
        marioai.Task.reset(self)
        rewards.Rewards.reset(self)


    def perform_action(self, action):
        marioai.Task.perform_action(self, action)
        rewards.Rewards.perform_action(self, action)


    def get_sensors(self):
        return rewards.Rewards.get_sensors(self)