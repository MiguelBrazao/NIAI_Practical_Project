import os


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

        # raw distance from the latest terminal FIT packet
        self.last_terminal_distance = 0.0

        # base terminal reward (distance * difficulty * ratio) WITHOUT kill multiplier,
        # stored so evaluate_agent can apply the kill bonus after potentially
        # discarding kills counted during a death sub-episode.
        self.base_terminal_reward = 0.0

        # cumulative kill counter across all sub-episodes of a single outer episode.
        # NOT reset between sub-episodes (level 0→1→2) so kills on earlier levels
        # continue to inflate the terminal reward on later levels. Reset explicitly
        # in evaluate_agent before the inner level loop, not inside reset().
        self.kill_count = 0

        # Optional kill-debug instrumentation. Enable with KILL_DEBUG=1.
        self.kill_debug = os.environ.get('KILL_DEBUG', '0').lower() in ('1', 'true', 'yes')
        self.kill_debug_counts = {
            'candidates': 0,
            'confirmed': 0,
        }

        # Touch grace window (in steps): direct stomps may disappear 1-2 frames
        # after contact rather than immediately on the next frame.
        self.recent_touch_frames = 0


    def reset(self):
        """
        Reset per-episode internal state.

        kill_count is intentionally not reset here because it accumulates across
        sub-episodes inside one outer evaluation episode.
        """
        self.last_sense = None
        # kill_count is NOT reset here so kills accumulate across sub-episodes
        # within the same outer episode. It is reset explicitly in evaluate_agent.
        self.last_terminal_distance = 0.0
        self.base_terminal_reward = 0.0
        for k in self.kill_debug_counts:
            self.kill_debug_counts[k] = 0
        self.recent_touch_frames = 0


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

        For terminal FIT packets, compute the distance-based terminal reward
        scaled by level difficulty and kill count. Any kills counted during a
        sub-episode where Mario dies are discarded by evaluate_agent before
        the terminal reward is applied.
        """
        sense = self.env.get_sensors()
        self.reward = 0.0

        # Fitness packet (no level scene)
        if sense.level_scene is None:
            self.status = sense.status
            self.last_terminal_distance = float(getattr(sense, 'distance', 0.0) or 0.0)

            # Base distance reward WITHOUT kill multiplier. The kill bonus is
            # applied later in evaluate_agent, after potentially discarding kills
            # counted during a death sub-episode (death-by-enemy triggers the same
            # touch+disappear signal as a stomp and would otherwise be a false kill).
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

            if self.kill_debug:
                dbg = self.kill_debug_counts
                print(
                    f"[KILL DEBUG SUMMARY] candidates={dbg['candidates']} "
                    f"confirmed={dbg['confirmed']} "
                    f"kill_count={self.kill_count}"
                )
        else:
            step_reward = self.compute_reward(sense, self.last_sense)
            self.reward = step_reward
            self.last_sense = sense

        return sense


    def kills(self, current_obs=None, last_obs=None):
        """
        Detect kills by tracking enemy touch + disappearance across frames.

        Primary path: a stompable enemy touches Mario in last_obs and is absent
        from current_obs (matched by type + proximity).
        Grace path: if disappearance is delayed 1-2 frames, recent_touch_frames
        keeps the candidate window open.
        Fallback path: if the tuple list misses it, a drop in the local stompable
        grid count after a recent touch counts as one kill.

        At most one kill is added per frame. Kills counted during a sub-episode
        where Mario dies are discarded by evaluate_agent after doEpisodes returns.
        """
        if current_obs is None or last_obs is None:
            return

        STOMPABLE = {2, 3, 4, 5, 6, 7, 13}

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

        def in_stomp_zone(enemy, mario_pos):
            """
            Touch zone around Mario in world coordinates.
            """
            if mario_pos is None:
                return False
            _, ex, ey = enemy
            mx, my = mario_pos
            dx = ex - mx
            dy = ey - my
            return (abs(dx) <= 24.0) and (-8.0 <= dy <= 48.0)

        def count_stompable_local(scene):
            """
            Count stompable cells in a local band around Mario in level_scene.
            """
            if scene is None:
                return 0
            r, c = 11, 11
            total = 0
            for dr in range(0, 4):
                rr = r + dr
                if rr < 0 or rr > 21:
                    continue
                for dc in range(-2, 3):
                    cc = c + dc
                    if cc < 0 or cc > 21:
                        continue
                    if scene[rr, cc] in STOMPABLE:
                        total += 1
            return total

        # Greedy one-to-one matching by same type and 2D displacement.
        MATCH_RADIUS = 24.0
        MATCH_RADIUS2 = MATCH_RADIUS * MATCH_RADIUS

        # Parse stompable enemies as (type, x, y) 
        # and match last->current by type+proximity.
        last_stomp = parse_stompable_enemies(last_obs)
        cur_stomp = parse_stompable_enemies(current_obs)
        mario_last = mario_xy(last_obs)

        last_scene = getattr(last_obs, 'level_scene', None)
        cur_scene = getattr(current_obs, 'level_scene', None)
        local_grid_drop = max(0, count_stompable_local(last_scene) - count_stompable_local(cur_scene))

        touching_last = [e for e in last_stomp if in_stomp_zone(e, mario_last)]
        if touching_last:
            self.recent_touch_frames = 2
        elif self.recent_touch_frames > 0:
            self.recent_touch_frames -= 1

        # Match each enemy in last_stomp to at most one in cur_stomp by type + proximity.
        # Unmatched enemies in last_stomp are "disappeared" this frame.
        used_cur = set()
        disappeared_last = []
        for lt, lx, ly in last_stomp:
            best_j = None
            best_d2 = None
            for cj, (ct, cx, cy) in enumerate(cur_stomp):
                if cj in used_cur or ct != lt:
                    continue
                d2 = (cx - lx) * (cx - lx) + (cy - ly) * (cy - ly)
                if d2 <= MATCH_RADIUS2 and (best_d2 is None or d2 < best_d2):
                    best_d2 = d2
                    best_j = cj
            if best_j is None:
                disappeared_last.append((lt, lx, ly))
            else:
                used_cur.add(best_j)

        # Candidates are disappeared enemies that were touching Mario in last_obs.
        # Also allow a short grace window after recent touch to catch stomps
        # whose disappearance is delayed by 1-2 frames.
        disappeared_touching = [e for e in disappeared_last if in_stomp_zone(e, mario_last)]
        if not disappeared_touching and self.recent_touch_frames > 0 and disappeared_last:
            disappeared_touching = [disappeared_last[0]]
        raw_candidate_count = len(disappeared_touching)

        # Fallback: if tuple disappearance did not fire but local stompable grid
        # count drops after a recent touch, treat as one kill candidate.
        if raw_candidate_count == 0 and self.recent_touch_frames > 0 and local_grid_drop > 0:
            raw_candidate_count = 1

        if raw_candidate_count > 0:
            self.kill_debug_counts['candidates'] += raw_candidate_count

            # At most one kill per frame to avoid bursty over-counting.
            self.kill_count += 1
            self.kill_debug_counts['confirmed'] += 1

            if self.kill_debug:
                print(
                    f"[KILL DEBUG] confirm(touch_disappear) raw_candidates={raw_candidate_count} "
                    f"kill_count={self.kill_count}"
                )