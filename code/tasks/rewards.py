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
        Stomp kill detection.

        Three conditions must hold simultaneously on the same frame:

        1. An enemy disappeared from the front zone (dx >= 0 in last frame, gone now).
        2. Mario just landed (was on_ground=False last frame, on_ground=True this frame).
        3. At least one enemy in the previous frame was within stomp range horizontally
           (|dx| <= 48 px ≈ 3 tiles).  Enemies that merely walk out of detection radius
           are at the far edge of the grid (~176 px away) and cannot satisfy this, which
           eliminates the main false-positive source (enemy disappears while Mario happens
           to be landing after jumping over a gap).
        """
        if current_obs is None or last_obs is None:
            return

        cur_enemies  = getattr(current_obs, 'enemies', []) or []
        last_enemies = getattr(last_obs,    'enemies', []) or []

        front_last = sum(1 for dx, dy, t in last_enemies if dx >= 0)
        front_cur  = sum(1 for dx, dy, t in cur_enemies  if dx >= 0)

        enemy_left_front = front_cur < front_last

        just_landed = (not getattr(last_obs, 'on_ground', True)
                       and getattr(current_obs, 'on_ground', False))

        # At least one enemy was close enough to stomp last frame
        _STOMP_RANGE_PX = 48
        enemy_in_range = any(abs(dx) <= _STOMP_RANGE_PX for dx, dy, t in last_enemies if dx >= 0)

        if enemy_left_front and just_landed and enemy_in_range:
            self.kill_count += front_last - front_cur