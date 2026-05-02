class Rewards:

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
        Kill detection via level_scene grid.

        Stompable enemy cells (Goomba 2/3, Koopa 4-7, Shell 13) appear in
        the 22x22 level_scene grid. An enemy that walks off the visible
        window also disappears from the grid, so a plain count-drop would
        produce false positives.

        Guard: a kill is only credited when Mario just landed
        (on_ground False->True) AND the stompable count in the grid dropped.
        Enemies that walk off screen don't cause a landing event, so they
        are correctly ignored.

        The full grid is counted (not just a window around Mario) so that
        enemies anywhere in the visible scene are tracked.
        """
        if current_obs is None or last_obs is None:
            return

        last_scene = getattr(last_obs,    'level_scene', None)
        cur_scene  = getattr(current_obs, 'level_scene', None)
        if last_scene is None or cur_scene is None:
            return

        STOMPABLE = {2, 3, 4, 5, 6, 7, 13}

        def count_stompable(scene):
            return int(sum((scene == v).sum() for v in STOMPABLE))

        prev_n = count_stompable(last_scene)
        curr_n = count_stompable(cur_scene)

        # DEBUG: print every frame where enemy cells are visible
        if prev_n > 0 or curr_n > 0:
            just_landed = (not getattr(last_obs, 'on_ground', True)
                           and getattr(current_obs, 'on_ground', False))
            print(f"[KILLS DEBUG] enemy cells: {prev_n} -> {curr_n}  "
                  f"just_landed={just_landed}  kill_count={self.kill_count}")

        just_landed = (not getattr(last_obs, 'on_ground', True)
                       and getattr(current_obs, 'on_ground', False))
        if just_landed and curr_n < prev_n:
            self.kill_count += prev_n - curr_n
            print(f"[KILLS DEBUG] KILL REGISTERED kill_count={self.kill_count}")