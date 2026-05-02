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
        Stomp kill detection using enemies list heuristics.

        Four conditions must all hold simultaneously:

        1. Mario just landed (on_ground False→True).
        2. A front enemy disappeared (front count dropped).
        3. At least one enemy in the previous frame was within horizontal
           stomp range (-8 <= dx <= 32 px). Negative dx tolerates a slight
           overshoot where Mario's centre has just passed over the enemy.
        4. That same enemy was at or below Mario (dy >= 0 px and dy <= 32 px).
           In MarioAI screen coords y increases downward, so dy >= 0 means
           the enemy is at or below Mario's centre — the stompable
           configuration. The former _MIN_BELOW_PX = 8 guard was causing
           false negatives on quick low-hop stomps where dy was only a few
           pixels; since just_landed + count-drop already constrain the
           detection well, dy >= 0 is safe here.
        """
        if current_obs is None or last_obs is None:
            return

        just_landed = (not getattr(last_obs, 'on_ground', True)
                       and getattr(current_obs, 'on_ground', False))
        if not just_landed:
            return

        last_enemies = getattr(last_obs,    'enemies', []) or []
        cur_enemies  = getattr(current_obs, 'enemies', []) or []

        _STOMP_RANGE_PX  = 16   # horizontal range ahead (~1 tile): realistic stomp contact zone
        _STOMP_BEHIND_PX = 8    # overshoot tolerance: Mario's centre may have just passed the enemy (half a tile behind)
        _MAX_BELOW_PX    = 16   # enemy must be within 1 tile below Mario's centre (stomp landing zone)

        # Enemy must be close horizontally (slight overshoot allowed) and at/below Mario
        stompable = any(-_STOMP_BEHIND_PX <= dx <= _STOMP_RANGE_PX and 0 <= dy <= _MAX_BELOW_PX
                        for dx, dy, t in last_enemies)
        if not stompable:
            return

        front_last = sum(1 for dx, dy, t in last_enemies if dx >= 0)
        front_cur  = sum(1 for dx, dy, t in cur_enemies  if dx >= 0)

        if front_cur < front_last:
            self.kill_count += front_last - front_cur