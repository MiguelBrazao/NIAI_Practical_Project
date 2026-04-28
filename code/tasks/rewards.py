import numpy as np

class Rewards:
    def __init__(
                self, 
                forward_reward_value=1.0,                       # base reward for moving forward (positive dx)
                backward_penalty_value=0.5,                     # base penalty for moving backward (negative dx)
                still_penalty_value=0.1,                        # penalty for not moving (dx = 0)
                direction_flip_penalty_value=-0.3,              # penalty for changing direction from forward to backward (only triggered when previously moving forward and now moving backward)
                direction_flip_extra_penalty_value=-0.1,        # additional penalty for consecutive direction flips (increases with each flip)
                direction_flip_extra_penalty_cap_value=-1.0,    # cap for the cumulative extra penalty from direction flips to prevent it from becoming too large
                jump_reward_value=0.2,                          # reward for jumping when it's beneficial (e.g., to survive or collect an item, determined by environment info) 
                jump_overhold_penalty=0.02,                     # penalty for holding the jump button too long without benefit (e.g., not making progress or getting stuck)
                jump_in_place_repeat_cap=-0.5,                  # cap for the cumulative penalty from jumping in place
                jump_in_place_dx_threshold=1.0,                 # threshold for considering a jump as "in place" (i.e., not making significant forward progress)
                jump_in_place_penalty=0.05,                     # incremental penalty for each consecutive jump in place (resets when a non-jump or a beneficial jump occurs)  
                jump_hold_min=3,                                # minimum consecutive jump presses to consider it a sustained jump (used to differentiate from a single jump press)
                jump_hold_max=10,                               # maximum consecutive jump presses to consider for sustained jump rewards/penalties (beyond this is likely a stuck jump and can be penalized)
                jump_hold_step_reward=0.05,                     # additional reward for each step of a sustained jump (encourages holding the jump button when beneficial)
                fall_threshold_value=-5,                        # threshold for detecting a fall (negative value, e.g., if Mario's y decreases by more than this value between steps, it's considered a fall)
                fall_penalty_value=0.5,                         # penalty for falling
                death_penalty_value=50.0,                       # penalty for dying (non-win status)
                finish_line_threshold=3000,                     # x position threshold for considering the level as completed (this can be tuned based on the level design)
                finish_line_bonus_value=500.0,                  # bonus for reaching the finish line
                jump_threshold=3.0,                             # threshold for determining a jump action based on the distance to the nearest relevant obstacle/enemy (can be tuned for different behaviors)
                shoot_threshold=5.0,                            # threshold for determining a shoot action based on the distance to the nearest enemy (can be tuned for different behaviors)
                duck_threshold=3.0,                             # threshold for determining a duck action based on the distance to the nearest obstacle/enemy (can be tuned for different behaviors)
                coin_reward_value=0.1,                          # small reward for collecting coins (encourages exploration and coin collection)
                power_up_reward_value=1.0,                      # reward for collecting power-ups (e.g., mushrooms, fire flowers)
                ):
        

        # Reward parameters (can be tuned for different behaviors)
        self.forward_reward_value = forward_reward_value
        self.backward_penalty_value = backward_penalty_value
        self.still_penalty_value = still_penalty_value
        self.direction_flip_penalty_value = direction_flip_penalty_value
        self.direction_flip_extra_penalty_value = direction_flip_extra_penalty_value
        self.direction_flip_extra_penalty_cap_value = direction_flip_extra_penalty_cap_value
        self.jump_reward_value = jump_reward_value
        self.jump_overhold_penalty = jump_overhold_penalty
        self.jump_in_place_repeat_cap = jump_in_place_repeat_cap
        self.jump_in_place_dx_threshold = jump_in_place_dx_threshold
        self.jump_in_place_penalty = jump_in_place_penalty
        self.jump_hold_min = jump_hold_min
        self.jump_hold_max = jump_hold_max
        self.jump_hold_step_reward = jump_hold_step_reward
        self.fall_threshold_value = fall_threshold_value
        self.fall_penalty_value = fall_penalty_value
        self.death_penalty_value = death_penalty_value
        self.finish_line_threshold = finish_line_threshold
        self.finish_line_bonus_value = finish_line_bonus_value
        self.jump_threshold = jump_threshold
        self.shoot_threshold = shoot_threshold
        self.duck_threshold = duck_threshold
        self.coin_reward_value = coin_reward_value
        self.power_up_reward_value = power_up_reward_value

        # Tracking variables for reward computation
        self.vars_current_obs = None
        self.vars_last_obs = None
        self.environment_info = None  # to store extracted environment info for potential later use
        self.cur_dx = 0
        self.prev_dx = 0
        self.direction_change_counter = 0
        self.last_action = None
        self.jump_press_counter = 0
        self.jump_in_place_counter = 0
        self.reward = 0.0


    def reset(self):
        self.prev_dx = 0
        self.cur_dx = 0
        self.direction_change_counter = 0
        self.last_action = None
        self.jump_press_counter = 0
        self.jump_in_place_counter = 0

    
    def perform_action(self, action):
        self.last_action = action
        if action is not None and len(action) > 3 and action[3] == 1:
            self.jump_press_counter += 1
        else:
            self.jump_press_counter = 0


    def get_sensors(self):
        """Override to apply final-episode penalties using the fitness packet (status).

        We avoid touching marioai/ and instead inspect the raw Observation from the
        environment here. For fitness packets (level_scene is None) we penalize
        non-win statuses and return the Observation as usual.
        """
        sense = self.env.get_sensors()

        # Fitness packet (no level scene)
        if sense.level_scene is None:
            # Base reward from distance
            self.reward = sense.distance
            self.status = sense.status

            # Penalize non-win endings (e.g., death)
            if self.status != 1:
                self.reward -= float(self.death_penalty_value)
            self.finished = True
        else:
            # Step reward computed by this task
            self.reward = self.compute_reward(sense, self.last_observation)
            self.last_observation = sense

        return sense
        

    def extract_environment(self, current_obs, last_obs):
        """
        Extracts relevant information from the current observation and computes:
        - categorized cells in front/behind (same as before)
        - nearest cell per category in front
        - nearest overall object in front
        - booleans: should_jump, should_run_shoot, should_duck (using separate thresholds)

        Parameters:
        - current_obs: current observation of the game state: dict_keys(['may_jump', 'on_ground', 'mario_pos', 'enemies', 'level_scene', 'status', 'distance', 'time_left', 'mario_mode', 'coins'])
        - last_obs: previous observation (can be None for the first step)
        """
        self.vars_current_obs = vars(current_obs)
        self.vars_last_obs = vars(last_obs) if last_obs is not None else None
        self.prev_dx = self.cur_dx
        self.reward = 0.0

        print("vars_current_obs keys:", self.vars_current_obs.keys())

        mario_position_in_world = (11, 11)  # Mario's position in the level grid (col,row)

        # cell value constants (same as before)
        soft_obstacle = -11
        hard_obstacle = -10
        empty_space = 0
        enemy_goomba = 2
        enemy_goomba_winged = 3
        enemy_red_koopa = 4
        enemy_red_koopa_winged = 5
        enemy_green_koopa = 6
        enemy_green_koopa_winged = 7
        enemy_bullet_bill = 8
        enemy_spiky = 9
        enemy_spiky_winged = 10
        enemy_piranha_flower = 12
        enemy_shell = 13
        item_mushroom = 14
        item_fire_flower = 15
        brick = 16
        enemy_obstacle = 20
        question_brick = 21
        mario_weapon_projectile = 25
        undefined = 44

        # prepare category lists
        soft_obstacles_in_front = []
        soft_obstacles_behind = []
        hard_obstacles_in_front = []
        hard_obstacles_behind = []
        empty_space_in_front = []
        empty_space_behind = []
        enemies_in_front = []
        enemies_behind = []
        enemy_obstacles_in_front = []
        enemy_obstacles_behind = []
        mushrooms_in_front = []
        mushrooms_behind = []
        fire_flowers_in_front = []
        fire_flowers_behind = []
        bricks_in_front = []
        bricks_behind = []
        question_bricks_in_front = []
        question_bricks_behind = []
        mario_weapon_projectiles_in_front = []
        mario_weapon_projectiles_behind = []
        undefined_in_front = []
        undefined_behind = []

        # iterate grid and populate front lists (only need front for decisions)
        for y in range(22):
            for x in range(22):
                cell_value = self.vars_current_obs['level_scene'][y, x]
                distance = ((x - mario_position_in_world[0]) ** 2 + (y - mario_position_in_world[1]) ** 2) ** 0.5
                cell = (cell_value, (x, y), distance)
                if x >= mario_position_in_world[0]:
                    if cell_value == soft_obstacle:
                        soft_obstacles_in_front.append(cell)
                    elif cell_value == hard_obstacle:
                        hard_obstacles_in_front.append(cell)
                    elif cell_value == empty_space:
                        empty_space_in_front.append(cell)
                    elif cell_value in {enemy_obstacle, enemy_goomba, enemy_goomba_winged, enemy_red_koopa, enemy_red_koopa_winged, enemy_green_koopa, enemy_green_koopa_winged, enemy_bullet_bill, enemy_spiky, enemy_spiky_winged, enemy_piranha_flower, enemy_shell}:
                        enemies_in_front.append(cell)
                    elif cell_value == item_mushroom:
                        mushrooms_in_front.append(cell)
                    elif cell_value == item_fire_flower:
                        fire_flowers_in_front.append(cell)
                    elif cell_value == brick:
                        bricks_in_front.append(cell)
                    elif cell_value == enemy_obstacle:
                        enemy_obstacles_in_front.append(cell)
                    elif cell_value == question_brick:
                        question_bricks_in_front.append(cell)
                    elif cell_value == mario_weapon_projectile:
                        mario_weapon_projectiles_in_front.append(cell)
                    else:
                        undefined_in_front.append(cell)
                else:
                    if cell_value == soft_obstacle:
                        soft_obstacles_behind.append(cell)
                    elif cell_value == hard_obstacle:
                        hard_obstacles_behind.append(cell)
                    elif cell_value == empty_space:
                        empty_space_behind.append(cell)
                    elif cell_value in {enemy_obstacle, enemy_goomba, enemy_goomba_winged, enemy_red_koopa, enemy_red_koopa_winged, enemy_green_koopa, enemy_green_koopa_winged, enemy_bullet_bill, enemy_spiky, enemy_spiky_winged, enemy_piranha_flower, enemy_shell}:
                        enemies_behind.append(cell)
                    elif cell_value == item_mushroom:
                        mushrooms_behind.append(cell)
                    elif cell_value == item_fire_flower:
                        fire_flowers_behind.append(cell)
                    elif cell_value == brick:
                        bricks_behind.append(cell)
                    elif cell_value == enemy_obstacle:
                        enemy_obstacles_behind.append(cell)
                    elif cell_value == question_brick:
                        question_bricks_behind.append(cell)
                    elif cell_value == mario_weapon_projectile:
                        mario_weapon_projectiles_behind.append(cell)
                    else:
                        undefined_behind.append(cell)

        # build environment_info based on categorized cells
        self.environment_info = {
            "in_front": {
                "soft_obstacles": soft_obstacles_in_front,
                "hard_obstacles": hard_obstacles_in_front,
                "empty_space": empty_space_in_front,
                "enemies": enemies_in_front,
                "mushrooms": mushrooms_in_front,
                "fire_flowers": fire_flowers_in_front,
                "bricks": bricks_in_front,
                "enemy_obstacles": enemy_obstacles_in_front,
                "question_bricks": question_bricks_in_front,
                "mario_weapon_projectiles": mario_weapon_projectiles_in_front,
                "undefined": undefined_in_front
            },
            "behind": {
                "soft_obstacles": soft_obstacles_behind,
                "hard_obstacles": hard_obstacles_behind,
                "empty_space": empty_space_behind,
                "enemies": enemies_behind,
                "mushrooms": mushrooms_behind,
                "fire_flowers": fire_flowers_behind,
                "bricks": bricks_behind,
                "enemy_obstacles": enemy_obstacles_behind,
                "question_bricks": question_bricks_behind,
                "mario_weapon_projectiles": mario_weapon_projectiles_behind,
                "undefined": undefined_behind
            }
        }

        # helper to get nearest in a list
        def nearest_cell(lst):
            if not lst:
                return None
            return min(lst, key=lambda c: c[2])

        # compute nearest per category
        nearest = {
            "soft_obstacles": nearest_cell(soft_obstacles_in_front),
            "hard_obstacles": nearest_cell(hard_obstacles_in_front),
            "empty_space": nearest_cell(empty_space_in_front),
            "enemies": nearest_cell(enemies_in_front),
            "enemy_obstacles": nearest_cell(enemy_obstacles_in_front),
            "mushrooms": nearest_cell(mushrooms_in_front),
            "mario_weapon_projectiles": nearest_cell(mario_weapon_projectiles_in_front),
        }

        # find overall closest object among selected categories
        candidates = [(k, v) for k, v in nearest.items() if v is not None]
        if candidates:
            closest_category, closest_cell = min(candidates, key=lambda kv: kv[1][2])
        else:
            closest_category, closest_cell = (None, None)

        # decide actions using separate thresholds
        closest_dist = closest_cell[2] if closest_cell is not None else float("inf")
        nearest_enemy = nearest["enemies"]
        nearest_enemy_dist = nearest_enemy[2] if nearest_enemy is not None else float("inf")
        nearest_projectile = nearest["mario_weapon_projectiles"]
        nearest_projectile_dist = nearest_projectile[2] if nearest_projectile is not None else float("inf")
        nearest_hard = nearest["hard_obstacles"]
        nearest_hard_dist = nearest_hard[2] if nearest_hard is not None else float("inf")

        # should_jump: nearest relevant obstacle/enemy within jump_threshold
        should_jump_flags = {"to_survive": False, "to_collect": False}
        if closest_category in {"enemies", "soft_obstacles", "hard_obstacles"} and closest_dist <= self.jump_threshold:
            should_jump_flags["to_survive"] = True
        if closest_category in {"mushrooms", "fire_flowers", "question_bricks"} and closest_dist <= self.jump_threshold:
            should_jump_flags["to_collect"] = True

        # should_run_shoot: enemy is the main trigger within shoot_threshold
        should_run_shoot = nearest_enemy_dist <= self.shoot_threshold

        # should_duck: projectile or low enemy close enough (projectile prioritized)
        # treat projectiles with y <= mario_y - 1 as duck candidates
        mario_y = mario_position_in_world[1]
        duck_candidate = False
        if nearest_projectile is not None and nearest_projectile_dist <= self.duck_threshold:
            px = nearest_projectile[1][0]; py = nearest_projectile[1][1]
            if py <= mario_y - 1:
                duck_candidate = True
        if nearest_enemy is not None and nearest_enemy_dist <= self.duck_threshold:
            ex = nearest_enemy[1][0]; ey = nearest_enemy[1][1]
            if ey <= mario_y - 1:
                duck_candidate = True
        should_duck = duck_candidate

        # attach computed nearest and decisions to environment_info
        self.environment_info["nearest"] = nearest
        self.environment_info["closest"] = (closest_category, closest_cell)
        self.environment_info["should_jump"] = should_jump_flags
        self.environment_info["should_run_shoot"] = should_run_shoot
        self.environment_info["should_duck"] = should_duck
        self.environment_info["thresholds"] = {
            "jump": self.jump_threshold,
            "shoot": self.shoot_threshold,
            "duck": self.duck_threshold
        }


    def forward_reward(self):
        """
        Computes the reward for the current state of the game based on Mario's actions 
        and the environment changes between the current and last observations.
        """
        # If we don't have a previous observation or mario position info, do nothing
        if self.vars_last_obs is None or self.vars_current_obs is None or self.vars_current_obs.get('mario_pos') is None or self.vars_last_obs.get('mario_pos') is None:
            return 0.0

        # store previous dx before updating
        self.prev_dx = self.cur_dx

        cur_x, _ = self.vars_current_obs['mario_pos']
        last_x, _ = self.vars_last_obs['mario_pos']
        dx_raw = cur_x - last_x

        # clamp dx to [-1, 1] (useful to avoid extreme step rewards)
        self.cur_dx = max(min(dx_raw, 1.0), -1.0)

        if self.cur_dx > 0:
            self.reward += self.forward_reward_value * self.cur_dx
        elif self.cur_dx < 0:
            self.reward -= self.backward_penalty_value * abs(self.cur_dx)
        else:
            self.reward -= self.still_penalty_value


    def erratic_movement_penalty(self):
        """
        Return a NEGATIVE penalty value (to be ADDED to the step reward by the caller).
        Penalize when previously moving forward and now moving backward.
        """
        penalty = 0.0

        # Penalize only when Mario was moving forward and now moves backward
        if self.prev_dx > 0 and self.cur_dx < 0:
            # Increment direction change counter and penalize
            self.direction_change_counter += 1
            penalty += self.direction_flip_penalty_value  # immediate penalty for turning backwards
            # Extra penalty for repeated backward turns in the same episode (apply cap)
            extra = self.direction_flip_extra_penalty_value * (self.direction_change_counter - 1)
            # extra is negative by config convention; ensure we don't go below the cap
            penalty += max(extra, self.direction_flip_extra_penalty_cap_value)
        
        # apply penalty to cumulative reward and return it
        self.reward += penalty


    def should_jump(self, type="to_survive"):
        """
        Determines whether Mario should jump based on the environment information extracted from the current observation. 
        This function can be used to inform the agent's decision-making process and encourage strategic use of jumping to navigate obstacles and enemies.
        """
        # Check if should jump to survive (avoid nearby threats)
        if type == "to_survive" and self.environment_info["should_jump"]["to_survive"]:
            return True
        
        if type == "to_collect" and self.environment_info["should_jump"]["to_collect"]:
            return True
        
        # Additional logic can be added here based on the specific environment information and desired behavior
        return False


    def jump_reward(self):
        """
        Computes a reward for upward movement (jumping) as a positive signal.
        Ensures the press counter is started only when Mario was on the ground in the previous observation.
        """
        if self.vars_last_obs is None or self.vars_current_obs['mario_pos'] is None or self.vars_last_obs['mario_pos'] is None:
            return
        
        _, cur_y = self.vars_current_obs['mario_pos']
        _, last_y = self.vars_last_obs['mario_pos']
        last_on_ground = bool(self.vars_last_obs.get('on_ground', True))
        cur_on_ground = bool(self.vars_current_obs.get('on_ground', True))

        # Only reward a beneficial jump, never penalize for not jumping
        if self.should_jump(type="to_survive") and cur_y > last_y:
            self.reward += self.jump_reward_value
        
        if last_on_ground and not cur_on_ground:
            self.jump_press_counter = 1
            self.jump_in_place_counter = 0
        elif not cur_on_ground and self.jump_press_counter > 0:
            self.jump_press_counter += 1
            if self.jump_hold_min <= self.jump_press_counter <= self.jump_hold_max:
                self.reward += self.jump_hold_step_reward
            elif self.jump_press_counter > self.jump_hold_max:
                over = self.jump_press_counter - self.jump_hold_max
                self.reward -= self.jump_overhold_penalty * over
            cur_x, _ = self.vars_current_obs['mario_pos']
            last_x, _ = self.vars_last_obs['mario_pos']
            if abs(cur_x - last_x) < self.jump_in_place_dx_threshold:
                self.jump_in_place_counter += 1
                penalty = self.jump_in_place_penalty * self.jump_in_place_counter
                self.reward -= min(penalty, abs(self.jump_in_place_repeat_cap))
            else:
                self.jump_in_place_counter = 0
        else:
            self.jump_press_counter = 0
            self.jump_in_place_counter = 0


    def fall_penalty(self):
        """
        Return a NEGATIVE penalty (to be ADDED to the step reward) when Mario falls
        by more than `abs(fall_threshold)` units between last_obs and current_obs.
        """
        if self.vars_last_obs is None or self.vars_current_obs.get('mario_pos') is None or self.vars_last_obs.get('mario_pos') is None:
            return

        _, cur_y = self.vars_current_obs['mario_pos']
        _, last_y = self.vars_last_obs['mario_pos']

        # If Mario dropped more than the threshold (threshold is negative by default),
        # return the negative penalty so the caller can add it to the reward.
        if cur_y < last_y + self.fall_threshold_value:
            self.reward -= float(self.fall_penalty_value)


    def coin_reward(self):
        """
        Computes a reward for collecting coins, which encourages the agent to explore and gather resources in the level.
        """
        if self.vars_last_obs is None or self.vars_current_obs.get('coins') is None or self.vars_last_obs.get('coins') is None:
            return
        
        cur_coins = self.vars_current_obs['coins']
        last_coins = self.vars_last_obs['coins']

        if cur_coins > last_coins:
            # Reward for each new coin collected (can be tuned)
            self.reward += float(self.coin_reward_value) * (cur_coins - last_coins)

    
    def power_up_reward(self):
        """
        Computes a reward for collecting power-ups (e.g., mushrooms, fire flowers), which encourages the agent to seek out and utilize power-ups for enhanced abilities.
        """
        if self.vars_last_obs is None or self.vars_current_obs.get('mario_mode') is None or self.vars_last_obs.get('mario_mode') is None:
            return
        
        cur_mode = self.vars_current_obs['mario_mode']
        last_mode = self.vars_last_obs['mario_mode']

        if cur_mode > last_mode:
            # Reward for powering up (can be tuned based on the mode increase)
            self.reward += float(self.power_up_reward_value) * (cur_mode - last_mode)
        elif cur_mode < last_mode:
            # Optional: small penalty for losing power-up status (e.g., getting hit)
            self.reward -= float(self.power_up_reward_value) * (last_mode - cur_mode)

    '''
    def enemy_kill_reward(self):
        """
        Computes a reward for defeating enemies, which encourages the agent to engage in combat and eliminate threats in the level.
        This can be implemented by tracking the number of enemies in the current and last observations and rewarding based on the decrease.
        """
        if self.vars_last_obs is None or self.vars_current_obs.get('enemies') is None or self.vars_last_obs.get('enemies') is None:
            return
        
        cur_enemies = self.vars_current_obs['enemies']
        last_enemies = self.vars_last_obs['enemies']

        # Count how many enemies were defeated (assuming enemy count decreases when defeated)
        if cur_enemies < last_enemies:
            # Reward for each enemy defeated (can be tuned)
            self.reward += float(self.enemy_kill_reward_value) * (last_enemies - cur_enemies)   
    '''

    def finish_line_bonus(self):
        """
        Computes a bonus for reaching or surpassing a certain x position, which approximates the end of the level. 
        This encourages the agent to complete the level rather than just surviving or engaging in combat.
        """
        if self.vars_current_obs['mario_pos'] is None:
            return
        
        cur_x, _ = self.vars_current_obs['mario_pos']

        # Large bonus for finishing the level (approximate end condition)
        if cur_x >= self.finish_line_threshold:
            self.reward += float(self.finish_line_bonus_value)