class Rewards:
    def __init__(
        self, 
        kills_reward_value=500.0,           # reward for defeating enemies (encourages combat and threat elimination)
        kills_threshold_min=0.0,            # minimum distance in world pixels for a kill to be valid (0 = stomp/contact kills count)
        kills_threshold_stomp=64.0,         # max distance for stomp kills — used when no fireball is on screen (~4 tiles)
        kills_threshold_fireball=200.0,     # max distance for fireball kills — used only when a fireball tile (25) is present in level_scene (~12 tiles)
        progression_reward_ratio=1.0,       # ratio to scale the distance reward (can be tuned to balance with other rewards and ensure it dominates as the primary objective)
    ):
        self.kills_reward_value = kills_reward_value
        self.kills_threshold_min = kills_threshold_min
        self.kills_threshold_stomp = kills_threshold_stomp
        self.kills_threshold_fireball = kills_threshold_fireball
        self.progression_reward_ratio = progression_reward_ratio

        self.last_sense = None                                  # to store the last observation for reward comparison (e.g., to compute movement, coin collection, enemy kills)
        self.vars_current_obs = None                            # to store the current observation variables for easy access (e.g., position, coins, enemies) and to avoid repeated unpacking during reward calculations
        self.vars_last_obs = None                               # to store the last observation variables for comparison with current observation in reward calculations (e.g., to compute movement, coin collection, enemy kills)
        self.reward = 0.0                                       # to store the computed reward for the current step, which can be accessed by the task's get_sensors method to return as part of the fitness packet
        self.kill_count = 0                                     # diagnostic counter: total enemies killed this episode (reset each episode)      
        self.use_progression = False                            # to track whether we should check the distance for a terminal reward in get_sensors, which can help ensure we apply the finish line bonus correctly when the finish line is reached (status == 1) and we have a valid distance measurement, while avoiding issues with distance being 0 or None in some edge cases (e.g., if the episode ends due to time running out or other non-finish-line reasons)
        self.mario_position_when_he_doesnt_kill = None          # to track Mario's position when progression rewards are stopped due to a nearby enemy, which can help prevent false positives in the kills() method when checking for enemy proximity to determine whether a kill is valid (i.e., we only want to stop forward rewards if an enemy is close enough to plausibly be a threat, and we want to check for kills based on whether an enemy was close in the last step when the enemy count dropped, rather than just checking the current step which could lead to false positives if an enemy walked off screen)

        self.FIREBALL_TILE = 25                                 # tile code for fireball in level_scene, used to determine whether to apply the wider kill threshold for fireball kills


    def reset(self):
        """
        Resets the internal state of the reward system. This method is called at the beginning of each episode
        to ensure that reward calculations start fresh and are not influenced by the previous episode's state.
        """
        self.last_sense = None
        self.kill_count = 0
        self.mario_position_when_he_doesnt_kill = None


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
            if self.use_progression:
                if self.mario_position_when_he_doesnt_kill is not None:
                    # If progression rewards were stopped due to a nearby enemy, apply a terminal reward based on how far Mario got before that happened, which encourages killing nearby enemies rather than just avoiding them and stopping progression rewards.
                    terminal_reward = (self.mario_position_when_he_doesnt_kill[0] * (1 + self.level_difficulty)) * self.progression_reward_ratio  # scale distance reward by level difficulty and distance reward ratio to encourage progress more in harder levels, even if the agent got stopped by a nearby enemy before reaching the finish line
                else:
                    terminal_reward = (sense.distance * (1 + self.level_difficulty)) * self.progression_reward_ratio  # scale distance reward by level difficulty and distance reward ratio to encourage progress more in harder levels

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


    def progression(self):
        """
        Computes a reward based on the distance traveled, which encourages the agent to
          make progress towards completing the level. This can be used as a terminal reward when 
          the episode ends (e.g., when Mario reaches the finish line or dies) to reflect how 
          close the agent was to finishing the level.
        """
        self.use_progression = True  # set flag to check distance in get_sensors, where we have access to the status and can apply the reward properly as a terminal reward


    # We should check enemy count change for a reward, but we should be careful about enemies walking off screen (which can cause false positives).
    def kills(self):
        """
        Rewards Mario for killing an enemy.
        An enemy is considered killed (rather than just walked off screen) if the enemy
        count dropped AND at least one enemy in the previous step was within the interval
        [kills_threshold_min, kills_threshold_max] world pixels of Mario.
        - kills_threshold_min=0 includes stomp/contact kills (Mario directly on enemy).
        - kills_threshold_stomp is the max range when no fireball is on screen (~4 tiles).
        - kills_threshold_fireball is the max range when Mario's fireball sprite is visible in the last frame.
        - Enemies that walked off screen (~256px+) are excluded by both thresholds.
        """
        if self.vars_last_obs is None or self.vars_current_obs is None:
            return

        cur_enemies = self.vars_current_obs.get('enemies') or []
        last_enemies = self.vars_last_obs.get('enemies') or []
        cur_count = len(cur_enemies)
        last_count = len(last_enemies)

        # Determine effective max threshold: wide range only when Mario's fireball is visible in level_scene
        _tmin2 = self.kills_threshold_min ** 2
        cur_scene  = self.vars_current_obs.get('level_scene')
        last_scene = self.vars_last_obs.get('level_scene')
        has_fireball_cur  = cur_scene  is not None and (cur_scene  == self.FIREBALL_TILE).any()
        has_fireball_last = last_scene is not None and (last_scene == self.FIREBALL_TILE).any()
        _tmax2_cur  = (self.kills_threshold_fireball if has_fireball_cur  else self.kills_threshold_stomp) ** 2
        _tmax2_last = (self.kills_threshold_fireball if has_fireball_last else self.kills_threshold_stomp) ** 2

        # Enemy is considered "close" for gating purposes based on current observation (X only). 
        # If he goes through an enemy that is close in the current step without killing it, we stop progression 
        # rewards until he gets a kill or the episode resets, which can help encourage combat rather than just avoidance.
        enemy_is_close = any(
            _tmin2 <= dx ** 2 <= _tmax2_cur
            for dx, dy, *_ in cur_enemies
        )

        if cur_count >= last_count:
            # No kill this step: if enemy is close, stop progression rewards until episode reset or until he gets a 
            # kill (which means he successfully dealt with the nearby threat rather than just avoiding it and getting 
            # stopped by it, which can help encourage combat rather than just avoidance). We track Mario's position when 
            # this happens to be able to apply a terminal reward based on how far he got if he never manages to kill 
            # that nearby enemy and the episode ends (e.g., due to time running out or dying), which can help 
            # encourage killing nearby enemies rather than just avoiding them and stopping progression rewards.
            if enemy_is_close:
                self.mario_position_when_he_doesnt_kill = self.vars_current_obs.get('mario_position')
            return  # no enemies killed this step

        # Check whether any enemy in last step was within the valid kill range (X only)
        enemy_was_close = any(
            _tmin2 <= dx ** 2 <= _tmax2_last
            for dx, dy, *_ in last_enemies
        )

        if enemy_was_close:
            n_kills = last_count - cur_count
            self.kill_count += n_kills
            self.reward += float(self.kills_reward_value) * n_kills * (1 + self.level_difficulty)  # scale kill reward by level difficulty to encourage killing more in harder levels, which can help prevent the agent from just rushing to the finish line and ignoring enemies in harder levels where they are more of a threat
            self.mario_position_when_he_doesnt_kill = None  # reset position tracking for stopping progression rewards since we got a valid kill this step, which means the enemy was close and we successfully killed it, so we can allow progression rewards again until/unless we encounter another nearby enemy without killing it