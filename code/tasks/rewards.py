class Rewards:

    def __init__(
        self,
        progression_reward_ratio=1.0,
        kill_multiplier=50.0,
    ):
        """Initialize reward and kill-tracking state for a task episode."""
        # Ratio to scale the distance reward
        self.progression_reward_ratio = progression_reward_ratio
        
        # Multiplicative bonus per kill: each kill scales the whole distance reward
        # by (1 + kill_count * kill_multiplier). Value chosen so that 1 kill at the
        # typical hunter distance (~340 units) beats the best pure runner
        # (max move_forward distance = 14208):  340 * (1 + 50) = 17340 > 14208.
        # With 0 kills the multiplier is 1.0, so MoveForwardTask (which never
        # calls kills()) degrades to pure distance reward.
        self.kill_multiplier = kill_multiplier

        # to store the last observation for reward comparison 
        # (e.g., to compute movement, coin collection, enemy kills)
        self.last_sense = None

        # to store the computed reward for the current step, which 
        # can be accessed by the task's get_sensors method to return 
        # as part of the fitness packet
        self.reward = 0.0  

        # base terminal reward (distance * difficulty * ratio) WITHOUT kill 
        # multiplier, stored so evaluate_agent can apply the kill bonus after 
        # settling sub_episode_touch_events into kill_count at sub-episode end.
        self.base_terminal_reward = 0.0

        # cumulative kill counter across all sub-episodes of a single outer episode.
        # NOT reset between sub-episodes (level 0→1→2) so kills on earlier levels
        # continue to inflate the terminal reward on later levels. Reset explicitly
        # in evaluate_agent before the inner level loop, not inside reset().
        self.kill_count = 0

        # Number of enemy-touch events detected in the current sub-episode.
        # Settled into kill_count by evaluate_agent at sub-episode end:
        # - WIN  -> all touch events are added to kill_count
        # - DEATH -> caller decides how many to keep (typically touches - 1)
        # Reset by reset() at the start of each sub-episode.
        self.sub_episode_touch_events = 0


    def reset(self):
        """
        Reset per-episode internal state.

        kill_count is intentionally not reset here because it accumulates across
        sub-episodes inside one outer evaluation episode.
        """
        self.last_sense = None
        # kill_count is NOT reset here so kills accumulate across sub-episodes
        # within the same outer episode. It is reset explicitly in evaluate_agent.
        self.base_terminal_reward = 0.0
        self.sub_episode_touch_events = 0


    def perform_action(self, action):
        """
        Hook called after each agent action.

        Kept for interface symmetry with Task; current reward logic does not
        require action-dependent state.
        """
        _ = action


    def get_sensors(self):
        """
        Fetch sensors and compute reward for either step packets or terminal FIT.

        For terminal FIT packets, stores the base distance reward (without kill
        multiplier) so evaluate_agent can apply the kill bonus after settling
        sub_episode_touch_events into kill_count.
        """
        sense = self.env.get_sensors()
        self.reward = 0.0

        # Fitness packet (no level scene)
        if sense.level_scene is None:
            self.status = sense.status

            # Base distance reward WITHOUT kill multiplier. The kill bonus is
            # applied later in evaluate_agent after sub_episode_touch_events
            # have been settled into kill_count.
            self.base_terminal_reward = (
                (sense.distance * (1 + self.level_difficulty)) * self.progression_reward_ratio
            )

            # assign (not +=): self.reward still holds last step's
            # value which was already counted by perform_action
            self.reward = self.base_terminal_reward

            # perform_action is skipped when
            # finished=True, so add directly
            self.cum_reward += self.reward
            self.finished = True
        else:
            step_reward = self.compute_reward(sense, self.last_sense)
            self.reward = step_reward
            self.last_sense = sense

        return sense


    def kills(self, last_obs=None):
        """
        Detect enemy contacts and accumulate them in sub_episode_touch_events.

        Each call checks whether any stompable enemy (by type ID) is within
        the touch zone of Mario's position in last_obs. Every new contact
        increments sub_episode_touch_events by one.

        sub_episode_touch_events is reset by reset() at the start of each
        sub-episode. kill_count accumulates across sub-episodes and is reset
        explicitly in evaluate_agent before the outer episode loop.
        """
        if last_obs is None:
            return

        # We remove the shell enemy type (13) from the stompable set 
        # because it behaves differently than other enemies.
        STOMPABLE = {2, 3, 4, 5, 6, 7} # {2, 3, 4, 5, 6, 7, 13}

        def parse_stompable_enemies(obs):
            """
            Parse stompable enemies as (type, x, y).
            """
            parsed = []
            for e in getattr(obs, 'enemies', []) or []:
                if not isinstance(e, (list, tuple)) or len(e) < 3:
                    continue
                try:
                    et = int(float(e[0]))
                    ex = float(e[1])
                    ey = float(e[2])
                except (TypeError, ValueError):
                    continue
                if et in STOMPABLE:
                    parsed.append((et, ex, ey))
            return parsed

        def mario_xy(obs):
            """
            Get Mario's (x, y) position from the observation.
            """
            mp = getattr(obs, 'mario_pos', None)
            if isinstance(mp, (list, tuple)) and len(mp) >= 2:
                try:
                    return float(mp[0]), float(mp[1])
                except (TypeError, ValueError):
                    return None
            return None

        def in_touch_zone(enemy, mario_pos):
            """
            Contact proxy around Mario in world coordinates.
            """
            if mario_pos is None:
                return False
            _, ex, ey = enemy
            mx, my = mario_pos
            dx = ex - mx
            dy = ey - my
            # Enemy is 16x16, so 12 gives some tolerance for 
            # partial overlaps and fast movement between frames.
            # Also, stomps reduce enemy height, so vertical tolerance
            # is especially important. This is a heuristic and may need tuning.
            return (abs(dx) <= 12.0) and (abs(dy) <= 12.0) 

        enemies = parse_stompable_enemies(last_obs)
        mario_last = mario_xy(last_obs)
        touched_enemy = any(in_touch_zone(e, mario_last) for e in enemies)

        if touched_enemy:
            self.sub_episode_touch_events += 1