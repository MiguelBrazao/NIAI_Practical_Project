class Rewards:
    def __init__(
                self, 

                controls_reward_value=1.0,                      # reward for taking an action (can be tuned to encourage more active behavior)
                controls_penalty_value=2.0,                     # penalty for pressing too many buttons at once (e.g., more than 2) to encourage more strategic and less erratic actions
                controls_button_threshold=3,                    # threshold for number of buttons pressed to start applying the controls penalty (e.g., 3 means penalty starts when pressing 3 or more buttons simultaneously)

                forward_reward_value=1.0,                       # base reward for moving forward (positive movement)
                backward_penalty_value=2.0,                     # base penalty for moving backward (negative movement)
                still_penalty_value=2.0,                        # penalty for not moving (movement = 0)

                jump_reward_value=1.0,                          # reward for jumping when it's beneficial (e.g., to survive or collect an item, determined by environment info) 
                jump_penalty_value=2.0,                         # penalty for jumping when it's not beneficial (e.g., unnecessary jump that could lead to danger, determined by environment info)
                jump_threshold=3.0,                             # threshold for determining a jump action based on the distance to the nearest relevant obstacle/enemy (can be tuned for different behaviors)

                duck_reward_value=1.0,                          # reward for ducking when it's beneficial (e.g., to avoid a projectile or low enemy, determined by environment info)
                duck_penalty_value=2.0,                         # penalty for ducking when it's not beneficial (e.g., unnecessary duck that could lead to danger, determined by environment info)
                duck_threshold=1.0,                             # threshold for determining a duck action based on the distance to the nearest obstacle/enemy (can be tuned for different behaviors)

                shoot_reward_value=1.0,                         # reward for shooting when it's beneficial (e.g., to defeat an enemy, determined by environment info)
                shoot_penalty_value=2.0,                        # penalty for shooting when it's not beneficial (e.g., unnecessary shooting that could lead to danger, determined by environment info)
                shoot_threshold=3.0,                            # threshold for determining a shoot action based on the distance to the nearest enemy (can be tuned for different behaviors)

                obstacles_reward_value=1.0,                     # reward for effectively navigating around obstacles (e.g., rewarding proximity to soft obstacles that can be passed and penalizing proximity to hard obstacles that should be avoided, determined by environment info)
                obstacles_penalty_value=1.0,                    # penalty for being close to hard obstacles or far from soft obstacles
                obstacles_threshold=3.0,                        # max grid-cell distance at which an obstacle is considered "close enough" to track for transposition

                coins_reward_value=1.0,                         # small reward for collecting coins (encourages exploration and coin collection)

                power_ups_reward_value=1.0,                     # reward for collecting power-ups (e.g., mushrooms, fire flowers)
                power_ups_penalty_value=2.0,                    # penalty for losing power-ups (e.g., going from fire flower to mushroom or small Mario)

                enemy_kills_reward_value=1.0,                   # reward for defeating enemies (encourages combat and threat elimination)
                
                death_penalty_value=100.0,                      # penalty for dying (can be tuned to balance with other rewards)                
                
                terminal_distance_scale=1.0,                    # divisor applied to sense.distance in the terminal reward (e.g. 16.0 converts world-pixels to grid cells, bringing it into the same scale as per-step rewards)
                ):
        
        # Reward parameters (can be tuned for different behaviors)
        self.controls_reward_value = controls_reward_value
        self.controls_penalty_value = controls_penalty_value
        self.controls_button_threshold = controls_button_threshold

        self.forward_reward_value = forward_reward_value
        self.backward_penalty_value = backward_penalty_value
        self.still_penalty_value = still_penalty_value

        self.jump_reward_value = jump_reward_value
        self.jump_penalty_value = jump_penalty_value
        self.jump_threshold = jump_threshold

        self.duck_reward_value = duck_reward_value
        self.duck_penalty_value = duck_penalty_value
        self.duck_threshold = duck_threshold

        self.shoot_reward_value = shoot_reward_value
        self.shoot_penalty_value = shoot_penalty_value
        self.shoot_threshold = shoot_threshold

        self.obstacles_reward_value = obstacles_reward_value
        self.obstacles_penalty_value = obstacles_penalty_value
        self.obstacles_threshold = obstacles_threshold

        self.coins_reward_value = coins_reward_value

        self.power_ups_reward_value = power_ups_reward_value
        self.power_ups_penalty_value = power_ups_penalty_value

        self.enemy_kills_reward_value = enemy_kills_reward_value

        self.death_penalty_value = death_penalty_value
        
        self.terminal_distance_scale = terminal_distance_scale

        # Tracking variables for reward computation
        self.last_action = None             # [backward, forward, crouch, jump, speed/bombs]
        self.last_sense = None              # to store the last observation for reward comparison (e.g., to compute movement, coin collection, enemy kills)

        self.vars_current_obs = None        # to store the current observation variables for easy access (e.g., position, coins, enemies) and to avoid repeated unpacking during reward calculations
        self.vars_last_obs = None           # to store the last observation variables for comparison with current observation in reward calculations (e.g., to compute movement, coin collection, enemy kills)
        self.environment_info = None        # to store extracted environment info for potential later use
        self.last_environment_info = None   # environment info from the previous step — used by shoot/duck rewards to match the state that prompted last_action
        
        self.reward = 0.0                   # to store the computed reward for the current step, which can be accessed by the task's get_sensors method to return as part of the fitness packet
        self.finish_line_reached = False    # to track if the finish line bonus has been awarded to prevent multiple bonuses


    def reset(self):
        self.last_action = None 
        self.last_sense = None              
        
        self.vars_current_obs = None
        self.vars_last_obs = None
        self.environment_info = None
        self.last_environment_info = None
        
        self.reward = 0.0
        self.finish_line_reached = False


    def perform_action(self, action):
        self.last_action = action 


    def get_sensors(self):
        """
        Override to apply final-episode rewards using the fitness packet (status).
        This allows to reward reaching the finish line and penalize dying in a way that is properly reflected in the fitness evaluation, since the episode ends immediately after these events and we won't have another step to apply those rewards/penalties.

        We avoid touching marioai/ and instead inspect the raw Observation from the
        environment here. For fitness packets (level_scene is None) we penalize
        non-win statuses and return the Observation as usual.
        """
        sense = self.env.get_sensors()

        # Fitness packet (no level scene)
        if sense.level_scene is None:
            self.status = sense.status
            terminal_reward = sense.distance / float(self.terminal_distance_scale)
            if self.status != 1:
                terminal_reward -= float(self.death_penalty_value)
            self.reward = terminal_reward
            self.cum_reward += terminal_reward  # perform_action is skipped when finished=True, so add directly
            self.finished = True
        else:
            step_reward = self.compute_reward(sense, self.last_sense)
            self.reward = step_reward
            self.last_sense = sense

        return sense
        

    def environment(self, current_obs, last_obs):
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
        self.last_environment_info = self.environment_info  # save before overwriting: aligns with last_action (chosen based on the previous obs)
        self.vars_current_obs = vars(current_obs)
        self.vars_last_obs = vars(last_obs) if last_obs is not None else None
        self.reward = 0.0

        mario_position_in_grid = (11, 11)  # Mario's position in the level grid (col,row)

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
        soft_obstacles_above = []
        soft_obstacles_below = []
        hard_obstacles_in_front = []
        hard_obstacles_behind = []
        hard_obstacles_above = []
        hard_obstacles_below = []
        empty_space_in_front = []
        empty_space_behind = []
        empty_space_above = []
        empty_space_below = []
        enemies_in_front = []
        enemies_behind = []
        enemies_above = []
        enemies_below = []
        enemy_obstacles_in_front = []
        enemy_obstacles_behind = []
        enemy_obstacles_above = []
        enemy_obstacles_below = []
        mushrooms_in_front = []
        mushrooms_behind = []
        mushrooms_above = []
        mushrooms_below = []
        fire_flowers_in_front = []
        fire_flowers_behind = []
        fire_flowers_above = []
        fire_flowers_below = []
        bricks_in_front = []
        bricks_behind = []
        bricks_above = []
        bricks_below = []
        question_bricks_in_front = []
        question_bricks_behind = []
        question_bricks_above = []
        question_bricks_below = []
        mario_weapon_projectiles_in_front = []
        mario_weapon_projectiles_behind = []
        mario_weapon_projectiles_above = []
        mario_weapon_projectiles_below = []
        undefined_in_front = []
        undefined_behind = []
        undefined_above = []
        undefined_below = []
        walls_in_front = []
        walls_behind = []
        walls_above = []
        walls_below = []

        # iterate grid and categorize cells into in_front / behind / above / below
        for y in range(22):
            for x in range(22):
                cell_value = self.vars_current_obs['level_scene'][y, x]
                distance = ((x - mario_position_in_grid[0]) ** 2 + (y - mario_position_in_grid[1]) ** 2) ** 0.5
                cell = (cell_value, (x, y), distance)
                if x > mario_position_in_grid[0]:
                    if cell_value == soft_obstacle:
                        soft_obstacles_in_front.append(cell)
                    elif cell_value == hard_obstacle:
                        hard_obstacles_in_front.append(cell)
                    elif cell_value == empty_space:
                        empty_space_in_front.append(cell)
                    elif cell_value in {enemy_goomba, enemy_goomba_winged, enemy_red_koopa, enemy_red_koopa_winged, enemy_green_koopa, enemy_green_koopa_winged, enemy_bullet_bill, enemy_spiky, enemy_spiky_winged, enemy_piranha_flower, enemy_shell}:
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
                    elif cell_value == undefined:
                        undefined_in_front.append(cell)
                    else:  # unknown cell value = wall/boundary
                        walls_in_front.append(cell)
                elif x < mario_position_in_grid[0]:
                    if cell_value == soft_obstacle:
                        soft_obstacles_behind.append(cell)
                    elif cell_value == hard_obstacle:
                        hard_obstacles_behind.append(cell)
                    elif cell_value == empty_space:
                        empty_space_behind.append(cell)
                    elif cell_value in {enemy_goomba, enemy_goomba_winged, enemy_red_koopa, enemy_red_koopa_winged, enemy_green_koopa, enemy_green_koopa_winged, enemy_bullet_bill, enemy_spiky, enemy_spiky_winged, enemy_piranha_flower, enemy_shell}:
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
                    elif cell_value == undefined:
                        undefined_behind.append(cell)
                    else:  # unknown cell value = wall/boundary
                        walls_behind.append(cell)
                else:  # x == mario_position_in_grid[0]: same column — above or below Mario
                    above = y < mario_position_in_grid[1]
                    if cell_value == soft_obstacle:
                        (soft_obstacles_above if above else soft_obstacles_below).append(cell)
                    elif cell_value == hard_obstacle:
                        (hard_obstacles_above if above else hard_obstacles_below).append(cell)
                    elif cell_value == empty_space:
                        (empty_space_above if above else empty_space_below).append(cell)
                    elif cell_value in {enemy_goomba, enemy_goomba_winged, enemy_red_koopa, enemy_red_koopa_winged, enemy_green_koopa, enemy_green_koopa_winged, enemy_bullet_bill, enemy_spiky, enemy_spiky_winged, enemy_piranha_flower, enemy_shell}:
                        (enemies_above if above else enemies_below).append(cell)
                    elif cell_value == item_mushroom:
                        (mushrooms_above if above else mushrooms_below).append(cell)
                    elif cell_value == item_fire_flower:
                        (fire_flowers_above if above else fire_flowers_below).append(cell)
                    elif cell_value == brick:
                        (bricks_above if above else bricks_below).append(cell)
                    elif cell_value == enemy_obstacle:
                        (enemy_obstacles_above if above else enemy_obstacles_below).append(cell)
                    elif cell_value == question_brick:
                        (question_bricks_above if above else question_bricks_below).append(cell)
                    elif cell_value == mario_weapon_projectile:
                        (mario_weapon_projectiles_above if above else mario_weapon_projectiles_below).append(cell)
                    elif cell_value == undefined:
                        (undefined_above if above else undefined_below).append(cell)
                    else:  # unknown cell value = wall/boundary
                        (walls_above if above else walls_below).append(cell)

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
                "undefined": undefined_in_front,
                "walls": walls_in_front
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
                "undefined": undefined_behind,
                "walls": walls_behind
            },
            "above": {
                "soft_obstacles": soft_obstacles_above,
                "hard_obstacles": hard_obstacles_above,
                "empty_space": empty_space_above,
                "enemies": enemies_above,
                "mushrooms": mushrooms_above,
                "fire_flowers": fire_flowers_above,
                "bricks": bricks_above,
                "enemy_obstacles": enemy_obstacles_above,
                "question_bricks": question_bricks_above,
                "mario_weapon_projectiles": mario_weapon_projectiles_above,
                "undefined": undefined_above,
                "walls": walls_above
            },
            "below": {
                "soft_obstacles": soft_obstacles_below,
                "hard_obstacles": hard_obstacles_below,
                "empty_space": empty_space_below,
                "enemies": enemies_below,
                "mushrooms": mushrooms_below,
                "fire_flowers": fire_flowers_below,
                "bricks": bricks_below,
                "enemy_obstacles": enemy_obstacles_below,
                "question_bricks": question_bricks_below,
                "mario_weapon_projectiles": mario_weapon_projectiles_below,
                "undefined": undefined_below,
                "walls": walls_below
            }
        }

        # helper to get nearest in a list
        def nearest_cell(lst):
            if not lst:
                return None
            return min(lst, key=lambda c: c[2])

        # compute nearest per category (in front)
        nearest = {
            "soft_obstacles": nearest_cell(soft_obstacles_in_front),
            "hard_obstacles": nearest_cell(hard_obstacles_in_front),
            "empty_space": nearest_cell(empty_space_in_front),
            "enemies": nearest_cell(enemies_in_front),
            "mushrooms": nearest_cell(mushrooms_in_front),
            "fire_flowers": nearest_cell(fire_flowers_in_front),
            "bricks": nearest_cell(bricks_in_front),
            "enemy_obstacles": nearest_cell(enemy_obstacles_in_front),
            "question_bricks": nearest_cell(question_bricks_in_front),
            "mario_weapon_projectiles": nearest_cell(mario_weapon_projectiles_in_front),
            "undefined": nearest_cell(undefined_in_front),
            "walls": nearest_cell(walls_in_front)
        }

        # compute nearest per category (above)
        nearest_above = {
            "soft_obstacles": nearest_cell(soft_obstacles_above),
            "hard_obstacles": nearest_cell(hard_obstacles_above),
            "empty_space": nearest_cell(empty_space_above),
            "enemies": nearest_cell(enemies_above),
            "mushrooms": nearest_cell(mushrooms_above),
            "fire_flowers": nearest_cell(fire_flowers_above),
            "bricks": nearest_cell(bricks_above),
            "enemy_obstacles": nearest_cell(enemy_obstacles_above),
            "question_bricks": nearest_cell(question_bricks_above),
            "mario_weapon_projectiles": nearest_cell(mario_weapon_projectiles_above),
            "undefined": nearest_cell(undefined_above),
            "walls": nearest_cell(walls_above)
        }

        # compute nearest per category (below)
        nearest_below = {
            "soft_obstacles": nearest_cell(soft_obstacles_below),
            "hard_obstacles": nearest_cell(hard_obstacles_below),
            "empty_space": nearest_cell(empty_space_below),
            "enemies": nearest_cell(enemies_below),
            "mushrooms": nearest_cell(mushrooms_below),
            "fire_flowers": nearest_cell(fire_flowers_below),
            "bricks": nearest_cell(bricks_below),
            "enemy_obstacles": nearest_cell(enemy_obstacles_below),
            "question_bricks": nearest_cell(question_bricks_below),
            "mario_weapon_projectiles": nearest_cell(mario_weapon_projectiles_below),
            "undefined": nearest_cell(undefined_below),
            "walls": nearest_cell(walls_below)
        }

        # find overall closest object among selected categories
        candidates = [(k, v) for k, v in nearest.items() if v is not None]
        if candidates:
            closest_category, closest_cell = min(candidates, key=lambda kv: kv[1][2])
        else:
            closest_category, closest_cell = (None, None)

        # decide actions using separate thresholds
        mario_y = mario_position_in_grid[1]
        nearest_enemy = nearest["enemies"]
        nearest_enemy_dist = nearest_enemy[2] if nearest_enemy is not None else float("inf")
        nearest_enemy_above = nearest_above["enemies"]
        nearest_enemy_below = nearest_below["enemies"]

        # should_jump (to_survive):
        # - enemy in front at same level or below mario (y >= mario_y) within jump_threshold
        # - enemy directly below within jump_threshold
        # - terrain in front (hard/soft obstacle, bricks, question bricks, walls, enemy obstacles) at Mario's row or
        #   above within jump_threshold (ground tiles at y > mario_y excluded to avoid always-True false positives)
        should_jump_flags = {"to_survive": False, "to_collect": False}

        enemy_in_front_low = (
            nearest_enemy is not None and
            nearest_enemy[2] <= self.jump_threshold and
            nearest_enemy[1][1] >= mario_y  # at same row or below mario
        )
        enemy_directly_below = (
            nearest_enemy_below is not None and
            nearest_enemy_below[2] <= self.jump_threshold
        )
        # obstacle_in_front: any terrain cell in front at Mario's row or above within jump_threshold.
        # Scans the raw per-category lists rather than nearest[cat] so a closer ground tile of the
        # same category cannot mask a level-blocking obstacle at Mario's height.
        # Excludes empty_space (air) and y > mario_y (ground tiles below Mario's feet).
        # Includes enemy_obstacles (value=20) — solid objects Mario must jump over.
        obstacle_in_front = any(
            cell[2] <= self.jump_threshold and cell[1][1] <= mario_y
            for lst in (
                soft_obstacles_in_front, hard_obstacles_in_front,
                bricks_in_front, question_bricks_in_front,
                walls_in_front, enemy_obstacles_in_front,
            )
            for cell in lst
        )
        if enemy_in_front_low or enemy_directly_below or obstacle_in_front:
            should_jump_flags["to_survive"] = True

        # should_jump (to_collect):
        # - mushroom or fire flower in front within jump_threshold
        # - mushroom or fire flower directly above within jump_threshold
        collectible_in_front = any(
            nearest[cat] is not None and nearest[cat][2] <= self.jump_threshold
            for cat in {"mushrooms", "fire_flowers"}
        )
        collectible_above = any(
            nearest_above[cat] is not None and nearest_above[cat][2] <= self.jump_threshold
            for cat in {"mushrooms", "fire_flowers"}
        )
        if collectible_in_front or collectible_above:
            should_jump_flags["to_collect"] = True

        # should_run_shoot: enemy in front within shoot_threshold
        should_run_shoot = nearest_enemy_dist <= self.shoot_threshold

        # should_duck:
        # - enemy in front AND above mario's row (y < mario_y) within duck_threshold (flying enemy approaching)
        # - enemy directly above within duck_threshold
        enemy_in_front_high = (
            nearest_enemy is not None and
            nearest_enemy_dist <= self.duck_threshold and
            nearest_enemy[1][1] < mario_y  # above mario's row
        )
        enemy_above = (
            nearest_enemy_above is not None and
            nearest_enemy_above[2] <= self.duck_threshold
        )
        should_duck = enemy_in_front_high or enemy_above

        # attach computed nearest and decisions to environment_info
        self.environment_info["nearest"] = nearest
        self.environment_info["nearest_above"] = nearest_above
        self.environment_info["nearest_below"] = nearest_below
        self.environment_info["closest"] = (closest_category, closest_cell)
        self.environment_info["should_jump"] = should_jump_flags
        self.environment_info["should_run_shoot"] = should_run_shoot
        self.environment_info["should_duck"] = should_duck
        self.environment_info["thresholds"] = {
            "jump": self.jump_threshold,
            "shoot": self.shoot_threshold,
            "duck": self.duck_threshold
        }


    def controls(self):
        """
        Computes reward if action array is not pressing on too many buttons at once (e.g., more than 2), which encourages more strategic and less erratic actions. 
        Computes penalty for pressing too many buttons simultaneously or none at all, which encourages the agent to focus on more effective actions rather than random button mashing.
        """
        if self.last_action is None:
            return
        
        buttons_pressed = sum(self.last_action)
        if buttons_pressed > self.controls_button_threshold:
            self.reward -= float(self.controls_penalty_value) * (buttons_pressed - self.controls_button_threshold)  # penalty for each button pressed beyond the threshold
        else:
            if buttons_pressed > 0:
                # Penalize backwards + forward together or crouch + jump together as they are typically ineffective or counterproductive combinations (can be tuned based on observed behavior)
                if (len(self.last_action) > 1 and self.last_action[0] == 1 and self.last_action[1] == 1) or (len(self.last_action) > 2 and self.last_action[2] == 1 and self.last_action[3] == 1):
                    self.reward -= float(self.controls_penalty_value)  # penalty for pressing backward and forward together
                else:
                    # All other combinations within the threshold get a small reward to encourage taking actions (can be tuned to encourage more active behavior)
                    self.reward += float(self.controls_reward_value) * buttons_pressed  # small reward for taking actions (can be tuned to encourage more active behavior)
                # Give an extra reward for pressing the speed button (index 4) when should_run_shoot is True, to encourage using speed/shoot strategically when there are nearby enemies
                if len(self.last_action) > 4 and self.last_action[4] == 1:
                    self.reward += float(self.controls_reward_value) # extra reward for speeding
            else:
                # No buttons pressed - apply a penalty to encourage taking actions (can be tuned to encourage more active behavior)
                self.reward -= float(self.controls_penalty_value)  # penalty for not taking any action (encourages more active behavior)

    
    def forward(self):
        """
        Computes a reward for forward movement and a penalty for backward movement or staying still, which encourages the agent to make progress through the level while discouraging regression and inactivity.
        """
        # If we don't have a previous observation or mario position info, do nothing
        if self.vars_last_obs is None or self.vars_current_obs is None or self.vars_current_obs.get('mario_pos') is None or self.vars_last_obs.get('mario_pos') is None:
            return

        cur_x, _ = self.vars_current_obs['mario_pos']
        last_x, _ = self.vars_last_obs['mario_pos']
        cur_movement = cur_x - last_x

        if cur_movement > 0:
            self.reward += self.forward_reward_value
        elif cur_movement < 0:
            self.reward -= self.backward_penalty_value
        else:
            self.reward -= self.still_penalty_value


    def jump(self, to_collect=False):
        """
        Computes a reward for upward movement (jumping) as a positive signal when Mario should jump.
        Computes a penalty for not jumping when Mario should have jumped, or for jumping when it wasn't necessary, as a negative signal to encourage more strategic use of jumping.
        """
        if self.vars_last_obs is None or self.vars_current_obs is None or ...:
            return

        def should_jump(to_collect=False):
            """
            Determines whether Mario should jump based on the environment information extracted from the current observation. 
            This function can be used to inform the agent's decision-making process and encourage strategic use of jumping to navigate obstacles and enemies.
            """
            if self.environment_info is None:
                return False
            if not to_collect and self.environment_info["should_jump"]["to_survive"]:
                return True
            if to_collect and self.environment_info["should_jump"]["to_collect"]:
                return True
            return False

        _, cur_y = self.vars_current_obs['mario_pos']
        _, last_y = self.vars_last_obs['mario_pos']

        if should_jump(to_collect=to_collect) and cur_y > last_y:
            # Reward for jumping when it's beneficial (e.g., to avoid enemies or gaps, determined by environment info) and actually moving upwards, which encourages the agent to use jumping strategically for survival and item collection.
            self.reward += self.jump_reward_value
        elif should_jump(to_collect=to_collect) and cur_y <= last_y:
            # Penalty for not jumping when it would be beneficial (e.g., failing to jump to avoid an enemy or gap, determined by environment info), which encourages the agent to recognize and act on opportunities to jump that could lead to better outcomes.
            self.reward -= self.jump_penalty_value
        elif not should_jump(to_collect=to_collect) and cur_y > last_y:
            # Penalty for jumping when it's not beneficial (e.g., unnecessary jump that could lead to danger, determined by environment info) and actually moving upwards, which encourages the agent to avoid unnecessary jumps that could lead to negative consequences.
            self.reward -= self.jump_penalty_value


    def duck(self, allow_ducking=True):
        """
        Computes a reward for ducking when it's beneficial (e.g., to avoid a projectile or low enemy), which encourages the agent to use ducking strategically for defense.
        Or a penalty for ducking when it should have been used but wasn't, which encourages the agent to recognize and act on opportunities to duck that could lead to better outcomes.
        This can be implemented by checking if the duck action was taken and if there was a nearby threat that could be avoided by ducking.
        """
        if self.last_action is None or self.last_environment_info is None:
            return
        
        should_duck = self.last_environment_info["should_duck"]
        # nearest_threat: whichever enemy (in front high or above) triggered should_duck
        nearest_threat = (
            self.last_environment_info["nearest_above"]["enemies"] or
            self.last_environment_info["nearest"]["enemies"]
        )

        if allow_ducking:
            # If ducking is allowed, reward for ducking when it's beneficial and penalize for not ducking when it would be beneficial. If ducking is not allowed, penalize for ducking at all.
            if should_duck and nearest_threat is not None and nearest_threat[2] <= self.duck_threshold:
                if len(self.last_action) > 2 and self.last_action[2] == 1:
                    self.reward += float(self.duck_reward_value) # reward for ducking when it's beneficial (e.g., to avoid a projectile or low enemy, determined by environment info) and actually ducking, which encourages the agent to use ducking strategically for defense.
                else:                
                    self.reward -= float(self.duck_penalty_value)  # penalty for not ducking when it would be beneficial
        else:
            # Penalty for ducking at all when it's not allowed, which encourages the agent to avoid unnecessary ducking that could lead to negative consequences. In this task, we can set allow_ducking=False to discourage ducking since it's not relevant for moving forward and could lead to more erratic behavior.
            if (self.last_action is not None and len(self.last_action) > 2 and self.last_action[2] == 1):
                self.reward -= float(self.duck_penalty_value)  # penalty for ducking when it's not allowed


    def shoot(self):
        """
        Computes a reward for shooting when it's beneficial (e.g., to defeat an enemy or clear a path), which encourages the agent to use shooting strategically.
        Or a penalty for shooting when it should have been used but wasn't, which encourages the agent to recognize and act on opportunities to shoot that could lead to better outcomes.
        This can be implemented by checking if the shoot action was taken and if there was a nearby enemy that could be affected.
        """
        if self.last_action is None or self.last_environment_info is None:
            return
        
        should_shoot = self.last_environment_info["should_run_shoot"]
        nearest_enemy = self.last_environment_info["nearest"]["enemies"]

        # Reward for shooting when there's a valid target within shoot_threshold
        if should_shoot and nearest_enemy is not None and nearest_enemy[2] <= self.shoot_threshold:
            if len(self.last_action) > 4 and self.last_action[4] == 1:
                self.reward += float(self.shoot_reward_value)
            else:
                self.reward -= float(self.shoot_penalty_value)  # penalty for not shooting when it would be beneficial


    def obstacles(self):
        """
        Rewards Mario for transposing obstacles (passing them):
        - Object was in front and is now behind, above, or below: transposed → reward
        - Object was above or below and is now behind: transposed → reward
        - Passable object was in front and is still in front (not passed): → penalty

        Transposition is detected by estimating where the previously-tracked cell would
        appear in the current grid, based on Mario's world-coordinate movement.
        """
        if (
            self.last_environment_info is None or self.environment_info is None or
            self.vars_current_obs is None or self.vars_last_obs is None or
            self.vars_current_obs.get('mario_pos') is None or self.vars_last_obs.get('mario_pos') is None
        ):
            return

        mario_col = 11
        cell_size = 16.0  # world pixels per grid cell (standard Mario grid)

        cur_mx, _ = self.vars_current_obs['mario_pos']
        last_mx, _ = self.vars_last_obs['mario_pos']
        dx_cells = (cur_mx - last_mx) / cell_size  # positive when Mario moves forward (right)

        passable_cats = {"soft_obstacles", "empty_space", "bricks", "question_bricks", "walls", "enemy_obstacles"}
        all_cats = passable_cats | {"hard_obstacles"}

        # Case 1: obstacle was in front last step
        for cat in all_cats:
            last_front = self.last_environment_info["nearest"].get(cat)
            if last_front is None or last_front[2] > self.obstacles_threshold:
                continue
            gx_pred = last_front[1][0] - dx_cells  # predicted grid column this step
            if gx_pred < mario_col or abs(gx_pred - mario_col) < 0.5:
                # Predicted to be behind or at Mario's column (above/below) → transposed
                self.reward += float(self.obstacles_reward_value)
            else:
                # Still in front — only penalize for passable obstacles Mario should have jumped
                if cat in passable_cats:
                    self.reward -= float(self.obstacles_penalty_value)

        # Case 2: obstacle was directly above or below last step, check if now behind
        for cat in all_cats:
            for key in ("nearest_above", "nearest_below"):
                last_side = self.last_environment_info.get(key, {}).get(cat)
                if last_side is None or last_side[2] > self.obstacles_threshold:
                    continue
                # Above/below cells are at x == mario_col (11); predict shift
                gx_pred = mario_col - dx_cells
                if gx_pred < mario_col:
                    # Now behind Mario → transposed
                    self.reward += float(self.obstacles_reward_value)
                

    def coins(self):
        """
        Computes a reward for collecting coins, which encourages the agent to explore and gather resources in the level.
        """
        if self.vars_last_obs is None or self.vars_current_obs.get('coins') is None or self.vars_last_obs.get('coins') is None:
            return
        
        cur_coins = self.vars_current_obs['coins']
        last_coins = self.vars_last_obs['coins']

        if cur_coins > last_coins:
            # Reward for each new coin collected (can be tuned)
            self.reward += float(self.coins_reward_value) * (cur_coins - last_coins)
            # if cur_coins < last_coins: likely wrapped (got extra life), ignore


    def power_ups(self):
        """
        Computes a reward for collecting power-ups (e.g., mushrooms, fire flowers), which encourages the agent to seek out and utilize power-ups for enhanced abilities.
        """
        if self.vars_last_obs is None or self.vars_current_obs.get('mario_mode') is None or self.vars_last_obs.get('mario_mode') is None:
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
    def enemy_kills(self):
        """
        Computes a reward for defeating enemies via fireball or stomp.
        Uses kill-type signals to filter out false positives (e.g., enemies walking off screen):
        - Fireball kill: Mario had a projectile in front near an enemy in the previous step.
        - Stomp kill: Mario was falling (downward movement) and there was an enemy directly below in the previous step.
        """
        if self.vars_last_obs is None or self.vars_current_obs is None:
            return

        cur_enemies = self.vars_current_obs['enemies']
        last_enemies = self.vars_last_obs['enemies']
        cur_count = len(cur_enemies) if hasattr(cur_enemies, '__len__') else int(cur_enemies)
        last_count = len(last_enemies) if hasattr(last_enemies, '__len__') else int(last_enemies)

        if cur_count >= last_count:
            return  # no enemies killed this step

        killed = last_count - cur_count

        # Fireball kill: Mario had a projectile in front AND an enemy in front in the previous step
        fireball_kill = (
            self.last_environment_info is not None and
            self.last_environment_info.get("nearest") is not None and
            self.last_environment_info["nearest"]["mario_weapon_projectiles"] is not None and
            self.last_environment_info["nearest"]["enemies"] is not None
        )

        # Stomp kill: Mario was falling and there was an enemy directly below in the previous step
        stomp_kill = False
        if (
            self.last_environment_info is not None and
            self.vars_current_obs.get('mario_pos') is not None and
            self.vars_last_obs.get('mario_pos') is not None
        ):
            _, cur_y = self.vars_current_obs['mario_pos']
            _, last_y = self.vars_last_obs['mario_pos']
            mario_falling = cur_y < last_y  # downward movement (higher y = higher position in world coords)
            enemy_below_last = (
                self.last_environment_info.get("nearest_below", {}).get("enemies") is not None
            )
            stomp_kill = mario_falling and enemy_below_last

        if fireball_kill or stomp_kill:
            self.reward += float(self.enemy_kills_reward_value) * killed