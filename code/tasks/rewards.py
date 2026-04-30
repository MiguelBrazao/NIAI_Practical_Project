class Rewards:
    def __init__(
        self, 
        forward_reward_value=1.0,           # base reward for moving forward — primary objective, must dominate per-step signals
        jump_reward_value=1.0,              # reward for jumping when beneficial — clearing terrain is harder than walking, worth more than one forward step
        coins_reward_value=1.0,             # small reward for collecting coins (encourages exploration and coin collection)
        power_ups_reward_value=1.0,         # reward for collecting power-ups (e.g., mushrooms, fire flowers)
        power_ups_penalty_value=2.0,        # penalty for losing power-ups (e.g., going from fire flower to mushroom or small Mario)
        kills_reward_value=1.0,             # reward for defeating enemies (encourages combat and threat elimination)
        kills_threshold=48.0,               # distance threshold in world pixels to consider an enemy kill valid (prevents false positives from enemies walking off screen)
        death_penalty_value=100.0,          # penalty for dying (can be tuned to balance with other rewards)                
    ):
        self.forward_reward_value = forward_reward_value
        self.jump_reward_value = jump_reward_value
        self.coins_reward_value = coins_reward_value
        self.power_ups_reward_value = power_ups_reward_value
        self.power_ups_penalty_value = power_ups_penalty_value
        self.kills_reward_value = kills_reward_value
        self.kills_threshold = kills_threshold
        self.death_penalty_value = death_penalty_value

        self.last_sense = None              # to store the last observation for reward comparison (e.g., to compute movement, coin collection, enemy kills)
        self.vars_current_obs = None        # to store the current observation variables for easy access (e.g., position, coins, enemies) and to avoid repeated unpacking during reward calculations
        self.vars_last_obs = None           # to store the last observation variables for comparison with current observation in reward calculations (e.g., to compute movement, coin collection, enemy kills)
        self.reward = 0.0                   # to store the computed reward for the current step, which can be accessed by the task's get_sensors method to return as part of the fitness packet
        self.check_distance = False         # to track whether we should check the distance for a terminal reward in get_sensors, which can help ensure we apply the finish line bonus correctly when the finish line is reached (status == 1) and we have a valid distance measurement, while avoiding issues with distance being 0 or None in some edge cases (e.g., if the episode ends due to time running out or other non-finish-line reasons)
        self.check_death = False            # to track whether we should check for death in the current step, which can help prevent multiple death penalties if the agent remains dead for multiple steps without resetting (e.g., due to a bug or edge case in the environment)      


    def reset(self):
        """
        Resets the internal state of the reward system. This method is called at the beginning of each episode
        to ensure that reward calculations start fresh and are not influenced by the previous episode's state.
        """
        self.last_sense = None


    def perform_action(self, action):
        """
        Gets called every time the agent performs an action. 
        We can use this to track the last action taken by the agent, 
        which can be useful for computing rewards that depend on the 
        agent's behavior (e.g., rewarding jumps when they lead to 
        progress or penalizing actions that lead to negative 
        outcomes). By storing the last action, we can also 
        analyze action patterns and their impact on the 
        reward signal.
        """
        self.last_action = action


    def get_sensors(self):
        """
        This method runs every step and is responsible for returning the current observation of the game state as a dictionary. 
        We override it to compute rewards based on the current and last observations, as well as the last action taken by the agent. 
        This allows us to implement a rich reward system that can encourage complex behaviors by providing feedback on progress, 
        interactions with enemies, coin collection, power-up usage, and more. By computing the reward in get_sensors, we ensure 
        that it is included in the fitness packet returned to the evolutionary algorithm for proper fitness evaluation.

        Override to apply final-episode rewards using the fitness packet (status).
        This allows to reward reaching the finish line and penalize dying in a way that is properly reflected in the 
        fitness evaluation, since the episode ends immediately after these events and we won't have another step 
        to apply those rewards/penalties.

        We avoid touching marioai/ and instead inspect the raw Observation from the
        environment here. For fitness packets (level_scene is None) we penalize
        non-win statuses and return the Observation as usual.
        """
        sense = self.env.get_sensors()
        self.reward = 0.0

        # Fitness packet (no level scene)
        if sense.level_scene is None:
            self.status = sense.status
            
            """
            Terminal reward/penalty for reaching the finish line or dying.
            Applied only once when the status changes to finished (1) or dead (0).
            And we have a valid distance measurement for the finish line bonus. 
            This ensures that we properly reward reaching the finish line.
            Or penalize dying in the fitness evaluation.
            """
            terminal_reward = 0.0
            if self.check_distance:
                terminal_reward = sense.distance
            if self.check_death and self.status != 1:
                terminal_reward -= float(self.death_penalty_value)

            self.reward = terminal_reward  # assign (not +=): self.reward still holds last step's value which was already counted by perform_action
            self.cum_reward += self.reward  # perform_action is skipped when finished=True, so add directly
            self.finished = True
        else:
            step_reward = self.compute_reward(sense, self.last_sense)
            self.reward = step_reward
            self.last_sense = sense

        return sense
        
    
    def observations(self, current_obs, last_obs):
        """
        Extracts relevant information from the current observation. Necessary for most methods to compute 
        rewards based on changes in the game state (e.g., movement, coin collection, enemy interactions). 
        By storing the current and last observations in a structured way, we can easily compare them to 
        compute various reward components.

        Parameters:
        - current_obs: current observation of the game state: 
        dict_keys(['may_jump', 'on_ground', 'mario_pos', 'enemies', 'level_scene', 
        'status', 'distance', 'time_left', 'mario_mode', 'coins'])
        - last_obs: previous observation (can be None for the first step)
        """
        self.vars_current_obs = vars(current_obs) if current_obs is not None else None
        self.vars_last_obs = vars(last_obs) if last_obs is not None else None


    def distance(self):
        """
        Computes a reward based on the distance to the finish line, which encourages the agent to
          make progress towards completing the level. This can be used as a terminal reward when 
          the episode ends (e.g., when Mario reaches the finish line or dies) to reflect how 
          close the agent was to finishing the level.
        """
        self.check_distance = True  # set flag to check distance in get_sensors, where we have access to the status and can apply the reward properly as a terminal reward

    
    def forward(self):
        """
        Computes a reward for forward movement, which 
        encourages the agent to make progress through 
        the level.
        """
        # If we don't have a previous observation or mario position info, do nothing
        if self.vars_last_obs is None or self.vars_current_obs is None or self.vars_current_obs.get('mario_pos') is None or self.vars_last_obs.get('mario_pos') is None:
            return

        cur_x, _ = self.vars_current_obs['mario_pos']
        last_x, _ = self.vars_last_obs['mario_pos']
        cur_movement = cur_x - last_x

        if cur_movement > 0:
            self.reward += self.forward_reward_value


    def jumps(self):
        """
        Computes a reward for upward movement (jumping) 
        as a positive signal when Mario should jump.
        """
        if self.vars_last_obs is None or self.vars_current_obs is None or self.vars_current_obs.get('mario_pos') is None or self.vars_last_obs.get('mario_pos') is None:
            return

        _, cur_y = self.vars_current_obs['mario_pos']
        _, last_y = self.vars_last_obs['mario_pos']

        on_ground = self.vars_current_obs.get('on_ground', True)

        if cur_y > last_y and not on_ground:
            # Reward for being in the air and rising, which encourages the agent to jump when it's beneficial (e.g., to clear an obstacle or reach a power-up).
            self.reward += self.jump_reward_value


    def coins(self):
        """
        Computes a reward for collecting coins, which 
        encourages the agent to explore and gather 
        resources in the level.
        """
        if self.vars_last_obs is None or self.vars_current_obs is None or self.vars_current_obs.get('coins') is None or self.vars_last_obs.get('coins') is None:
            return
        
        cur_coins = self.vars_current_obs['coins']
        last_coins = self.vars_last_obs['coins']

        if cur_coins > last_coins:
            # Reward for each new coin collected (can be tuned)
            self.reward += float(self.coins_reward_value) * (cur_coins - last_coins)
            # if cur_coins < last_coins: likely wrapped (got extra life), ignore


    def power_ups(self):
        """
        Computes a reward for collecting power-ups (e.g., mushrooms, 
        fire flowers), which encourages the agent to seek out and 
        utilize power-ups for enhanced abilities.
        """
        if self.vars_last_obs is None or self.vars_current_obs is None or self.vars_current_obs.get('mario_mode') is None or self.vars_last_obs.get('mario_mode') is None:
            return
        
        cur_mode = self.vars_current_obs['mario_mode']
        last_mode = self.vars_last_obs['mario_mode']

        if cur_mode > last_mode:
            # Reward for powering up (can be tuned based on the mode increase)
            self.reward += float(self.power_ups_reward_value) * (cur_mode - last_mode)
        elif cur_mode < last_mode and cur_mode > 0:  # only penalize if still alive (mode > 0)
            self.reward -= float(self.power_ups_penalty_value) * (last_mode - cur_mode)
            # if cur_mode == 0: Mario is dead, death_penalty in get_sensors handles it


    # We should check enemy count change for a reward, but we should be careful about enemies walking off screen (which can cause false positives).
    def kills(self):
        """
        Rewards Mario for killing an enemy.
        An enemy is considered killed (rather than just walked off screen) if the enemy
        count dropped AND at least one enemy was within self.kills_threshold world pixels of
        Mario in the previous step (enemies are encoded as (dx, dy, type) offsets from Mario).
        """
        if self.vars_last_obs is None or self.vars_current_obs is None:
            return

        cur_enemies = self.vars_current_obs.get('enemies') or []
        last_enemies = self.vars_last_obs.get('enemies') or []
        cur_count = len(cur_enemies)
        last_count = len(last_enemies)

        if cur_count >= last_count:
            return  # no enemies killed this step

        # Check whether any enemy in last step was close enough to Mario to plausibly be killed
        enemy_was_close = any(
            (dx ** 2 + dy ** 2) <= self.kills_threshold ** 2
            for dx, dy, *_ in last_enemies
        )

        if enemy_was_close:
            self.reward += float(self.kills_reward_value) * (last_count - cur_count)

    
    def deaths(self):
        """
        Penalizes Mario for dying, which encourages the agent to 
        avoid dangerous situations and learn survival strategies.
        """
        self.check_death = True  # set flag to check for death in get_sensors, where we have access to the status and can apply the penalty properly as a terminal reward