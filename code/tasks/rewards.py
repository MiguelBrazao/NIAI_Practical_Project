class Rewards:

    def __init__(
        self,
        progression_reward_ratio=1.0,
        kill_multiplier=50.0,
    ):
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

        # cumulative kill counter across all sub-episodes of a single outer episode.
        # NOT reset between sub-episodes (level 0→1→2) so kills on earlier levels
        # continue to inflate the terminal reward on later levels. Reset explicitly
        # in evaluate_agent before the inner level loop, not inside reset().
        self.kill_count = 0

        # sticky flag: True once a stompable enemy enters the stomp window;
        # cleared after a kill or when Mario is safely back on the ground
        self.enemy_was_recently_close = False

        # frames since enemy_was_recently_close was last set; used to expire the
        # flag so that an enemy sighted near Mario but not stomped cannot cause a
        # false positive if it exits the screen many frames later.
        # A real stomp's count-drop arrives within ~6 frames of the last close sighting.
        self.frames_since_close = 0

        # sticky flag: True once Mario leaves the ground;
        # ensures kills are only credited for stomps, not ground collisions
        self.mario_was_airborne = False

        # counts consecutive frames Mario has been on the ground;
        # flags are only reset after 2 ground frames so the kill drop
        # (which arrives one frame after landing) is never missed
        self.frames_on_ground = 0


    def reset(self):
        """
        Resets the internal state of the reward system. This method is called 
        at the beginning of each episode to ensure that reward calculations 
        start fresh and are not influenced by the previous episode's state.
        """
        self.last_sense = None
        # kill_count is NOT reset here so kills accumulate across sub-episodes
        # within the same outer episode. It is reset explicitly in evaluate_agent.
        self.enemy_was_recently_close = False
        self.frames_since_close = 0
        self.mario_was_airborne = False
        self.frames_on_ground = 0


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
        Hybrid kill detection using both enemy tuples and level_scene counts.
        Increments self.kill_count, which accumulates across all sub-episodes
        (level 0→1→2) within a single outer episode. This means kills scored
        on earlier levels continue to inflate the terminal reward on later levels.
        kill_count is reset explicitly in evaluate_agent before the inner level
        loop, not inside task.reset().

        Tuple path (preferred when tuple telemetry is available):
            1) Parse stompable enemies from last/current observations as
               (type, x, y) in world coordinates.
            2) Match last->current by same type and nearest 2D displacement
               within MATCH_RADIUS (42 units).
            3) Unmatched last-frame enemies are "disappeared" this frame.
            4) Keep only disappeared enemies that were in the stomp zone
               around Mario in last_obs (|dx| <= 22, 0 <= dy <= 52).
            5) Confirm only if global grid stompable count also drops in the
               same frame; credited kills are min(tuple_disappeared, grid_drop).

        Grid fallback (only when tuple telemetry is absent):
            If enemy tuples are unavailable in both observations, register kills
            from a global stompable-cell drop gated by the sticky proximity flag
            (enemy_was_recently_close), which is set when a stompable enemy is in
            the local grid window below Mario (rows 12-13, cols 10-12).

        Both confirmation paths require:
            - mario_was_airborne: Mario left the ground at some point this jump,
              ruling out ground-level collisions.
            - mario_just_landed: Mario is on the ground or within the 2-frame
              post-landing window, ruling out enemies that fall into a pit while
              Mario is still high up mid-air.

        The sticky enemy_was_recently_close flag expires after CLOSE_EXPIRY (10)
        consecutive frames with no enemy in the grid window, preventing enemies
        that were briefly near Mario from causing false positives many frames later.
        """
        if current_obs is None or last_obs is None:
            return
        
        # Observation of the game state: 
        # 'may_jump', 'on_ground', 'mario_pos', 
        # 'enemies', 'level_scene', 'status', 
        # 'distance', 'time_left', 
        # 'mario_mode', 'coins'

        # Get the level_scene grids from the last and current observations. 
        # If either is missing, we cannot perform kill detection, so we return early. 
        last_scene = getattr(last_obs,    'level_scene', None)
        cur_scene  = getattr(current_obs, 'level_scene', None)
        if last_scene is None or cur_scene is None:
            return

        STOMPABLE = {2, 3, 4, 5, 6, 7, 13}

        def count_stompable(scene):
            """
            Counts the number of stompable enemies 
            in the given level_scene grid.
            """
            return int(sum((scene == v).sum() for v in STOMPABLE))

        r, c = 11, 11

        # Count stompable enemies in 
        # the last and current scenes.
        prev_n = count_stompable(last_scene)
        curr_n = count_stompable(cur_scene)

        # Get on_ground status for the current observation; 
        # default to False if not present.
        on_ground = getattr(current_obs, 'on_ground', False)

        # Update airborne flag BEFORE kill check so it's set on the kill frame
        if not on_ground:
            self.mario_was_airborne = True

        # Grid-side proximity (legacy guard): stompable enemy in a local window
        # below Mario in the previous frame.
        enemy_close_last_frame_grid = any(
            last_scene[r + dr, c + dc] in STOMPABLE
            for dr in range(1, 3)
            for dc in range(-1, 2)
        )
        
        def parse_stompable_enemies(obs):
            """
            Tuple-side refinement: parse stompable enemies as (type, x, y), then
            match last->current by enemy type + 2D distance. Unmatched last-frame
            enemies are "disappeared" this frame.
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
            Conservative stomp zone in world coords relative to Mario center.
            dy>0 means enemy is below Mario in screen/world coordinates.
            """
            if mario_pos is None:
                return False
            _, ex, ey = enemy
            mx, my = mario_pos
            dx = ex - mx
            dy = ey - my
            return (abs(dx) <= 22.0) and (0.0 <= dy <= 52.0)

        # Greedy one-to-one matching by same type and 2D displacement.
        # Slightly larger radius avoids false "disappear" events when an enemy
        # moves quickly between frames but is still alive.
        MATCH_RADIUS = 42.0
        MATCH_RADIUS2 = MATCH_RADIUS * MATCH_RADIUS

        # Parse stompable enemies as (type, x, y) 
        # and match last->current by type+proximity.
        last_stomp = parse_stompable_enemies(last_obs)
        cur_stomp = parse_stompable_enemies(current_obs)
        mario_last = mario_xy(last_obs)

        # Match each enemy in last_stomp to at most one in cur_stomp by type + proximity.
        # Unmatched enemies in last_stomp are "disappeared" this frame.
        used_cur = set()
        disappeared_last = []
        for li, (lt, lx, ly) in enumerate(last_stomp):
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

        # Count disappeared enemies that were close to Mario in the last frame as potential stomp kills.
        disappeared_close = [e for e in disappeared_last if in_stomp_zone(e, mario_last)]

        # Count the number of disappeared enemies that were close to Mario in the last frame.
        tuple_kill_count = len(disappeared_close)

        # Proximity check for grid-based kill detection: any enemy cell in the stomp window in the last frame.
        enemy_close_last_frame_tuple = any(in_stomp_zone(e, mario_last) for e in last_stomp)

        # Final proximity check: either the legacy grid-based check OR the refined 
        # tuple-based check indicates an enemy was close in the last frame.
        enemy_close_last_frame = enemy_close_last_frame_grid or enemy_close_last_frame_tuple

        # Update the sticky "recently close" flag: set to True if an enemy 
        # was close this frame, otherwise increment the expiry counter.
        if enemy_close_last_frame:
            self.enemy_was_recently_close = True
            self.frames_since_close = 0
        else:
            self.frames_since_close += 1

        # Expire the sticky flag if no enemy was seen in the window for too long.
        # A real stomp's count-drop arrives within ~6 frames of the last close sighting
        # (3 airborne + 1 landing + kill at frames_gnd=2 = ~6).  After 10 frames
        # without a close sighting, any remaining count drop is almost certainly an
        # enemy scrolling off-screen, not a stomp.
        CLOSE_EXPIRY = 10
        if self.frames_since_close > CLOSE_EXPIRY:
            self.enemy_was_recently_close = False

        cur_enemies = getattr(current_obs, 'enemies', [])
        last_enemies = getattr(last_obs, 'enemies', [])

        '''
        # DEBUG: print whenever enemies are visible
        if prev_n > 0 or curr_n > 0:
            print(f"[KILLS DEBUG] enemy cells: {prev_n} -> {curr_n}  "
                  f"recently_close={self.enemy_was_recently_close}  "
                  f"since_close={self.frames_since_close}  "
                  f"airborne={self.mario_was_airborne}  "
                  f"frames_gnd={self.frames_on_ground}  "
                  f"tuple_kills={tuple_kill_count}  "
                  f"kill_count={self.kill_count}  "
                  f"last enemies={last_enemies}  "
                  f"cur enemies={cur_enemies}")
        '''
                  
        # Kill check BEFORE any flag resets.
        # Requires ALL of:
        #   1. enemy was recently in stomp window (sticky, expires after 10 frames)
        #   2. the global stompable count dropped (enemy truly gone, not just moved)
        #   3. Mario was airborne at some point (rules out ground-level collisions)
        #   4. Mario has landed (on_ground OR within the 2-frame post-landing window):
        #      guards against an enemy falling into a pit while Mario is still
        #      high up mid-air — a stomp always ends with Mario touching ground
        mario_just_landed = on_ground or self.frames_on_ground > 0
        grid_drop_kill_count = max(0, prev_n - curr_n)

        # If tuple telemetry exists this frame, rely on tuple+grid agreement.
        # Only use the old sticky grid fallback when tuple telemetry is absent.
        tuple_tracking_available = bool(
            (getattr(last_obs, 'enemies', None) or [])
            or (getattr(current_obs, 'enemies', None) or [])
        )

        confirmed_kills = 0
        if tuple_tracking_available:
            # Confirm tuple disappearances only when the global stompable count
            # also drops in the same frame.
            if tuple_kill_count > 0 and grid_drop_kill_count > 0:
                confirmed_kills = min(tuple_kill_count, grid_drop_kill_count)
        elif grid_drop_kill_count > 0 and self.enemy_was_recently_close:
            # Legacy fallback path for builds/frames without enemy tuples.
            confirmed_kills = grid_drop_kill_count

        # Update kill count and reset flags if we registered a kill. 
        # Otherwise, if Mario is safely on the ground for 2+ frames without 
        # a close enemy, reset the sticky flags to avoid false positives from 
        # future enemy sightings.
        if confirmed_kills > 0 and self.mario_was_airborne and mario_just_landed:
            self.kill_count += confirmed_kills
            self.enemy_was_recently_close = False
            self.frames_since_close = 0
            self.mario_was_airborne = False
            self.frames_on_ground = 0
            # print(f"[KILLS DEBUG] KILL REGISTERED +{confirmed_kills} kill_count={self.kill_count}")
        elif on_ground and self.frames_on_ground >= 2:
            # 2+ consecutive ground frames without a kill — safe to reset sticky flags
            self.enemy_was_recently_close = False
            self.mario_was_airborne = False

        # Update frames_on_ground: increment if on_ground, reset to 0 if not.
        self.frames_on_ground = (self.frames_on_ground + 1) if on_ground else 0