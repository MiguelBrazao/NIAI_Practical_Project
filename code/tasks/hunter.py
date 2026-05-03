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
        Computes the reward for the current state of the game based on Mario's 
        actions and the environment changes between the current and last observations.
        This function evaluates Mario's progress, interactions with enemies, and overall 
        performance to calculate a reward value. The reward is used as the fitness function 
        for the evolutionary algorithm.

        Parameters:
        - current_obs: The current observation of the game state;
        - last_obs: The previous observation of the game state;

        Returns:
        - reward (float): The computed reward value based on the game state changes.

        Notes for Students:
        - This function is critical for defining the algorithm behavior. 
          The reward function directly impacts the fitness evaluation of the AI.
        - You are encouraged to edit and experiment with this function to design 
          a reward system that aligns with the objectives of the project.
        - Consider the balance between encouraging progress, rewarding kills, 
          and penalizing undesirable behaviors (e.g., cowardice or reckless actions).
        """
        # Detect enemy contacts and accumulate in sub_episode_touch_events;
        # evaluate_agent settles these into kill_count at sub-episode end.
        self.kills(last_obs=last_obs)
        
        # Return the computed reward and 
        # reset internal state for next step
        return self.reward


    def reset(self):
        """
        Resets the internal state of the reward system. This method is called 
        at the beginning of each episode to ensure that reward calculations 
        start fresh and are not influenced by the previous episode's state.
        """
        marioai.Task.reset(self)
        rewards.Rewards.reset(self)
        

    def get_sensors(self):
        """
        This method runs every step and is responsible for returning the current observation of 
        the game state as a dictionary. We override it to compute rewards based on the current and 
        last observations, as well as the last action taken by the agent. This allows us to implement 
        a rich reward system that can encourage complex behaviors by providing feedback on progress, 
        interactions with enemies, coin collection, power-up usage, and more. By computing the 
        reward in get_sensors, we ensure that it is included in the fitness packet returned 
        to the evolutionary algorithm for proper fitness evaluation.

        Override to apply final-episode rewards using the fitness packet (status). This 
        allows to reward reaching the finish line and penalize dying in a way that is properly
        reflected in the fitness evaluation, since the episode ends immediately after these 
        events and we won't have another step to apply those rewards/penalties.

        We avoid touching marioai/ and instead inspect the raw Observation from the
        environment here. For fitness packets (level_scene is None) we penalize
        non-win statuses and return the Observation as usual.
        """
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