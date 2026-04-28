import torch
import torch.nn as nn
import numpy as np
import marioai
from .utils import (
    erratic_movement_penalty,
    forward_reward,
    jump_reward,
    fall_penalty,
    finish_line_bonus,
)


class MoveForwardTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MoveForward"

        # Tracking variables for reward computation
        self.cur_dx = 0
        self.prev_dx = 0
        self.direction_change_counter = 0
        self.last_action = None
        self.jump_press_counter = 0
        self.jump_in_place_counter = 0


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

        reward, self.cur_dx = forward_reward(current_obs, last_obs)

        erratic_penalty, self.direction_change_counter = erratic_movement_penalty(self.cur_dx, self.prev_dx, self.direction_change_counter)
        reward -= erratic_penalty

        jump_reward_value, self.jump_press_counter, self.jump_in_place_counter = jump_reward(current_obs, last_obs, self.cur_dx, self.jump_press_counter, self.jump_in_place_counter)
        reward += jump_reward_value
             
        reward -= fall_penalty(current_obs, last_obs)

        reward += finish_line_bonus(current_obs, last_obs)

        return reward


    def reset(self):
        super().reset()
        self.prev_dx = 0
        self.direction_change_counter = 0
        self.last_action = None
        self.jump_press_counter = 0
        self.jump_in_place_counter = 0


    def perform_action(self, action):
        try:
            # Capture action to track jump holding, then forward to base implementation.
            self.last_action = action
            if action is not None and len(action) > 3 and action[3] == 1:
                self.jump_press_counter += 1
            else:
                self.jump_press_counter = 0
        except Exception:
            pass

        super().perform_action(action)


    def get_sensors(self):
        """Override to apply final-episode penalties using the fitness packet (status).

        We avoid touching marioai/ and instead inspect the raw Observation from the
        environment here. For fitness packets (level_scene is None) we penalize
        non-win statuses and return the Observation as usual.
        """
        sense = self.env.get_sensors()

        # Fitness packet (no level scene)
        if sense.level_scene is None:
            # Base reward from distance
            self.reward = sense.distance
            self.status = sense.status

            # Penalize non-win endings (e.g., death)
            if self.status != 1:
                self.reward -= 50.0
            self.finished = True
        else:
            # Step reward computed by this task
            self.reward = self.compute_reward(sense, self.last_observation)
            self.last_observation = sense

        return sense