class Rewards:
    # includes enemy_obstacle (20) to catch all enemy encodings, 
    # which is important for kill/slip-through detection in kills()
    _ENEMY_TILE_VALUES = frozenset({2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 20})  

    # Mario is always at the centre of the 22×22 level_scene grid (row 11, col 11)
    _MARIO_TILE = (11, 11)                                                      

    def __init__(
        self,
        # Ratio to scale the distance reward (can be tuned to balance 
        # with other rewards and ensure it dominates as the primary objective)
        progression_reward_ratio=1.0,
        # Multiplicative bonus per kill: each kill scales the whole distance reward
        # by (1 + kill_count * kill_multiplier). With 0 kills the multiplier is 1.0
        # so the reward degrades to pure distance (safe for MoveForwardTask which
        # never calls kills()).
        kill_multiplier=5.0,
    ):
        self.progression_reward_ratio = progression_reward_ratio
        self.kill_multiplier = kill_multiplier

        # to store the last observation for reward comparison 
        # (e.g., to compute movement, coin collection, enemy kills)
        self.last_sense = None

        # to store the computed reward for the current step, which 
        # can be accessed by the task's get_sensors method to return 
        # as part of the fitness packet
        self.reward = 0.0  

        # diagnostic counter: total enemies 
        # killed this episode (reset each episode)                                                           
        self.kill_count = 0


    def reset(self):
        """
        Resets the internal state of the reward system. This method is called 
        at the beginning of each episode to ensure that reward calculations 
        start fresh and are not influenced by the previous episode's state.
        """
        self.last_sense = None
        self.kill_count = 0


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
        sense = self.env.get_sensors()
        self.reward = 0.0

        # Fitness packet (no level scene)
        if sense.level_scene is None:
            self.status = sense.status
            
            # Distance reward scaled by level difficulty and progression ratio.
            # Multiplied by a kill bonus: each kill amplifies the whole reward,
            # so a killer that goes half as far still beats a runner with 0 kills
            # once kill_multiplier is tuned appropriately.
            # When kill_count=0 (e.g. MoveForwardTask never calls kills()),
            # the multiplier is exactly 1.0 — pure distance reward.
            terminal_reward = (
                (sense.distance * (1 + self.level_difficulty)) * self.progression_reward_ratio
                * (1 + self.kill_count * self.kill_multiplier)
            )

            # assign (not +=): self.reward still holds last step's
            # value which was already counted by perform_action
            self.reward = terminal_reward  

            # perform_action is skipped when 
            # finished=True, so add directly
            self.cum_reward += self.reward  
            self.finished = True
        else:
            step_reward = self.compute_reward(sense, self.last_sense)
            self.reward = step_reward
            self.last_sense = sense

        return sense


    def kills(self, current_obs=None, last_obs=None):
        """
        Kill detection via level_scene zone counting.

        The scene is divided into two zones relative to Mario's centre (col 11):
          front zone : cols >= MC  (ahead of Mario, including directly under him)
          behind zone: cols <  MC  (already passed)

        Detection uses _ENEMY_TILE_VALUES (not just stompable) to catch all enemy
        encodings, including enemy_obstacle (20), which would otherwise be missed.
        The enemies list from the server is NOT used here — it is capped at ~3
        entries and drops enemies that have passed behind Mario.

        KILL: the front-zone count dropped AND the behind-zone count did NOT rise
        → an enemy disappeared from the front zone without going behind Mario.
        → kill_count is incremented; a flat kill bonus is applied at the terminal step.
        """
        if current_obs is None or last_obs is None:
            return
        
        vars_current_obs = vars(current_obs) if current_obs is not None else None
        vars_last_obs = vars(last_obs) if last_obs is not None else None

        last_scene = vars_last_obs.get('level_scene')
        cur_scene  = vars_current_obs.get('level_scene')
        if last_scene is None or cur_scene is None:
            return

        # Mario always at col 11
        _, MC = self._MARIO_TILE  

        def _zone(scene, col_start, col_end):
            """
            Counts the number of enemy tiles in the specified zone of the level_scene.
            The zone is defined by the column range [col_start, col_end) across all rows
            """
            return sum(
                1
                for r in range(22)
                for c in range(col_start, col_end)
                if int(scene[r, c]) in self._ENEMY_TILE_VALUES
            )

        # Count enemies in front and behind 
        # zones for both last and current scenes
        front_last  = _zone(last_scene, MC, 22)
        front_cur   = _zone(cur_scene,  MC, 22)
        behind_last = _zone(last_scene, 0,  MC)
        behind_cur  = _zone(cur_scene,  0,  MC)

        # enemy(s) left the front zone without appearing behind Mario → killed
        enemy_left_front = front_cur < front_last
        enemy_slipped_past = enemy_left_front and behind_cur > behind_last
        confirmed_kill = enemy_left_front and not enemy_slipped_past

        if confirmed_kill:
            self.kill_count += front_last - front_cur