def vars_observation(current_obs, last_obs):
    """
    Converts the current and last observations into dictionaries of their attributes for easier access.
    This function is useful for extracting relevant information from the observations in a structured way, which can then be used for reward computation or other analyses.
    Parameters:
    - current_obs: The current observation of the game state;
    - last_obs: The previous observation of the game state;
    Returns:
    - A tuple containing two dictionaries: (vars_current_obs, vars_last_obs), where each dictionary contains the attributes of the respective observation.
    """
    vars_current = vars(current_obs)
    vars_last = vars(last_obs) if last_obs is not None else None
    return vars_current, vars_last


def environment_info(obs):
    """
    Extracts relevant information from the current observation to compute the reward.
    This function can be customized to include various features that are relevant for the task and 
    to shape the reward in a way that encourages desirable behaviors and discourages undesirable ones.
    Parameters:
    - obs: The current observation from the environment.
    Returns:
    - A dictionary containing relevant information extracted from the observations.
    """

    vars_current_obs, _ = vars_observation(obs, None)
    
    # Extract environment information for fitness function based on the full window
    mario_position_in_world = (11, 11)  # Mario's position in the world (column, row)
    
    soft_obstacle = -11
    hard_obstacle = -10
    empty_space = 0
    enemy_obstacle = 20
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

    soft_obstacles_in_front = []
    soft_obstacles_behind = []
    hard_obstacles_in_front = []
    hard_obstacles_behind = []
    empty_space_in_front = []
    empty_space_behind = []
    enemies_in_front = []
    enemies_behind = []
    mushrooms_in_front = []
    mushrooms_behind = []
    fire_flowers_in_front = []
    fire_flowers_behind = []  
    bricks_in_front = []
    bricks_behind = []
    enemy_obstacles_in_front = []
    enemy_obstacles_behind = []
    question_bricks_in_front = []
    question_bricks_behind = []
    mario_weapon_projectiles_in_front = []
    mario_weapon_projectiles_behind = []
    undefined_in_front = []
    undefined_behind = []

    for y in range(22):
        for x in range(22):                
            cell_value = vars_current_obs['level_scene'][y, x]

            # Calculate cell distance from Mario (at 11,11) to determine front/behind
            distance = ((x - mario_position_in_world[0]) ** 2 + (y - mario_position_in_world[1]) ** 2) ** 0.5

            cell = (cell_value, (x, y), distance)

            if cell_value == soft_obstacle:
                if x >= mario_position_in_world[0]:
                    soft_obstacles_in_front.append(cell)
                else:
                    soft_obstacles_behind.append(cell)
            elif cell_value == hard_obstacle:
                if x >= mario_position_in_world[0]:
                    hard_obstacles_in_front.append(cell)
                else:
                    hard_obstacles_behind.append(cell)
            elif cell_value == empty_space:
                if x >= mario_position_in_world[0]:
                    empty_space_in_front.append(cell)
                else:
                    empty_space_behind.append(cell)
            elif cell_value in {enemy_obstacle, enemy_goomba, enemy_goomba_winged, enemy_red_koopa, enemy_red_koopa_winged, enemy_green_koopa, enemy_green_koopa_winged, enemy_bullet_bill, enemy_spiky, enemy_spiky_winged, enemy_piranha_flower, enemy_shell}:
                if x >= mario_position_in_world[0]:
                    enemies_in_front.append(cell)
                else:
                    enemies_behind.append(cell)
            elif cell_value == item_mushroom:
                if x >= mario_position_in_world[0]:
                    mushrooms_in_front.append(cell)
                else:
                    mushrooms_behind.append(cell)
            elif cell_value == item_fire_flower:
                if x >= mario_position_in_world[0]:
                    fire_flowers_in_front.append(cell)
                else:
                    fire_flowers_behind.append(cell)
            elif cell_value == brick:
                if x >= mario_position_in_world[0]:
                    bricks_in_front.append(cell)
                else:
                    bricks_behind.append(cell)
            elif cell_value == enemy_obstacle:
                if x >= mario_position_in_world[0]:
                    enemy_obstacles_in_front.append(cell)
                else:
                    enemy_obstacles_behind.append(cell)
            elif cell_value == question_brick:
                if x >= mario_position_in_world[0]:
                    question_bricks_in_front.append(cell)
                else:
                    question_bricks_behind.append(cell)
            elif cell_value == mario_weapon_projectile:
                if x >= mario_position_in_world[0]:
                    mario_weapon_projectiles_in_front.append(cell)
                else:
                    mario_weapon_projectiles_behind.append(cell)
            elif cell_value == undefined:
                if x >= mario_position_in_world[0]:
                    undefined_in_front.append(cell)
                else:
                    undefined_behind.append(cell)

    # Save environment information for fitness function
    env_info = {
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

    return env_info


def forward_reward(current_obs, last_obs, forward_reward=1.0, backward_penalty=2.0, still_penalty=0.5):
    """
    Computes the reward for the current state of the game based on Mario's actions 
    and the environment changes between the current and last observations.
    This function evaluates Mario's progress, interactions with enemies, and overall 
    performance to calculate a reward value. The reward is used as the fitness function for the evolutionary algorithm.
    Parameters:
    - current_obs: The current observation of the game state;
    - last_obs: The previous observation of the game state;
    Returns:
    - reward (float): The computed reward value based on the game state changes.
    Notes for Students:
    - This function is critical for defining the algorithm behavior. The reward function 
      directly impacts the fitness evaluation of the AI.
    - You are encouraged to edit and experiment with this function to design a reward 
      system that aligns with the objectives of the project.
    - Consider the balance between encouraging progress, rewarding kills, and penalizing 
      undesirable behaviors (e.g., cowardice or reckless actions).
    """

    vars_current_obs, vars_last_obs = vars_observation(current_obs, last_obs)

    # last_obs is None on the very first step
    if vars_last_obs is None or vars_current_obs['mario_pos'] is None or vars_last_obs['mario_pos'] is None:
        return 0, 0

    reward = 0.0

    cur_x, _ = vars_current_obs['mario_pos']
    last_x, _ = vars_last_obs['mario_pos']

    # Delta movement along x
    dx = cur_x - last_x

    # Reward moving right (forward). Scale with dx (usually 0 or 1).
    if dx > 0:
        reward += forward_reward * dx
    elif dx < 0:
        # Penalize moving backwards more strongly
        reward -= backward_penalty * abs(dx)
    else:
        # No horizontal movement, could be neutral or slightly negative to encourage progress
        reward -= still_penalty

    return reward, dx


def erratic_movement_penalty(cur_dx, prev_dx, direction_change_counter, direction_flip_penalty=1.0, direction_flip_extra_penalty=0.5):
    """
    Computes an additional penalty for erratic movement, defined as frequent direction changes 
    between forward and backward movement. This encourages the agent to maintain consistent progress 
    rather than oscillating back and forth.
    Returns:
    - penalty (float): The computed penalty value based on the frequency of direction changes.
    Notes for Students:
    - This function can be integrated into the main reward function to provide a more comprehensive 
      evaluation of the agent's behavior. Consider how to track direction changes and how to scale 
      the penalty appropriately to discourage erratic behavior without overly punishing necessary adjustments.
    """
    # Erratic movement penalty: direction change between consecutive steps
    # (i.e., flipping from forward to backward or vice-versa)
    
    reward = 0.0

    # Penalize only when Mario was moving forward and now moves backward
    if prev_dx > 0 and cur_dx < 0:
        # Increment direction change counter and penalize
        direction_change_counter += 1
        reward -= direction_flip_penalty  # immediate penalty for turning backwards
        # Extra penalty for repeated backward turns in the same episode
        reward -= min(direction_flip_extra_penalty * (direction_change_counter - 1), 5.0)
    
    return reward, direction_change_counter


def should_jump(current_obs, distance_threshold=1.5):
    """
    Determines whether Mario should jump based on the environment information extracted from the current observation. 
    This function can be used to inform the agent's decision-making process and encourage strategic use of jumping to navigate obstacles and enemies.
    """
    env_info = environment_info(current_obs)
    
    # If there's an enemy or hard obstacle directly in front of Mario, consider jumping
    if env_info['in_front']['soft_obstacles'] or env_info['in_front']['hard_obstacles'] or env_info['in_front']['empty_space'] or env_info['in_front']['enemies'] or env_info['in_front']['enemy_obstacles']:
        # Check if they are close enough to warrant a jump (e.g., within 1 cell distance)
        for category in ['soft_obstacles', 'hard_obstacles', 'empty_space', 'enemies', 'enemy_obstacles']:
            for cell in env_info['in_front'][category]:
                _, (x, y), distance = cell
                if distance <= distance_threshold:  # Threshold for "close enough" to jump
                    return True
    
    # Additional logic can be added here based on the specific environment information and desired behavior

    return False


def jump_reward(current_obs, last_obs, cur_dx, jump_press_counter, jump_in_place_counter, jump_reward=0.5, jump_hold_step_reward=0.1, jump_hold_min=5, jump_hold_max=15, jump_overhold_penalty=0.05, jump_in_place_dx_threshold=0.5, jump_in_place_penalty=0.2, jump_in_place_repeat_cap=5):
    """
    Computes a reward for upward movement (jumping) as a positive signal. This encourages the agent to utilize jumping as a means of navigating the environment and overcoming obstacles.
    Returns:
    - reward (float): The computed reward value for jumping.
    Notes for Students:
    - This function can be integrated into the main reward function to provide an additional incentive for using jumps effectively. Consider how to detect jumps based on changes in the y position and how to scale the reward to encourage strategic use of jumping without promoting excessive or reckless jumping behavior.
    """

    reward = 0.0

    try:
        vars_current_obs, vars_last_obs = vars_observation(current_obs, last_obs)
        if vars_last_obs is None or vars_current_obs['mario_pos'] is None or vars_last_obs['mario_pos'] is None:
            return reward, jump_press_counter, jump_in_place_counter
        
        _, cur_y = vars_current_obs['mario_pos']
        _, last_y = vars_last_obs['mario_pos']

        # Reward upward movement (jumping) as a positive signal
        if should_jump(current_obs) and cur_y > last_y:
            reward += jump_reward
        else:
            # If Mario is not jumping when it should, consider a small penalty to encourage better timing
            reward -= jump_reward * 0.5  # Penalize missed jump opportunities

        # Use observation's on_ground flag to track jump hold duration and detect
        # jumps-in-place. If Mario is airborne but not progressing forward (dx small),
        # penalize as a discouraged behavior.
        # Jump hold reward/penalty (small positive per-step for held jumps within bounds)
        if not vars_current_obs['on_ground']:
            jump_press_counter += 1
            if jump_press_counter >= jump_hold_min and jump_press_counter <= jump_hold_max:
                reward += jump_hold_step_reward
            elif jump_press_counter > jump_hold_max:
                over = jump_press_counter - jump_hold_max
                reward -= jump_overhold_penalty * over

            # Penalize jumping in place: airborne but no (or negligible) forward progress
            # Threshold chosen small because dx is often 0 or 1 per step.
            if abs(cur_dx) < jump_in_place_dx_threshold:
                jump_in_place_counter += 1
                # increase penalty for repeated jump-in-place behaviour, capped
                reward -= min(jump_in_place_penalty * jump_in_place_counter, jump_in_place_repeat_cap)
        else:
            # landed -> reset counters
            jump_press_counter = 0
            jump_in_place_counter = 0
    except Exception:
        pass
    return reward, jump_press_counter, jump_in_place_counter


def fall_penalty(current_obs, last_obs, fall_threshold=-5, fall_penalty=5.0):
    """
    Computes a penalty for falling below a certain y threshold compared to the last observation. 
    This encourages the agent to avoid falling into pits or off platforms.
    Returns:
    - penalty (float): The computed penalty value based on the change in y position.
    Notes for Students:
    - This function can be integrated into the main reward function to provide a more comprehensive 
      evaluation of the agent's behavior. Consider how to set the fall threshold and penalty to effectively discourage falling without overly punishing necessary drops (e.g., dropping down from a platform).
    """
    try:
        vars_current_obs, vars_last_obs = vars_observation(current_obs, last_obs)
        if vars_last_obs is None or vars_current_obs['mario_pos'] is None or vars_last_obs['mario_pos'] is None:
            return 0
        
        _, cur_y = vars_current_obs['mario_pos']
        _, last_y = vars_last_obs['mario_pos']

        # Penalize falling below a certain threshold
        if cur_y < last_y + fall_threshold:
            return fall_penalty
    except Exception:
        pass
    return 0


def finish_line_bonus(current_obs, last_obs, level_complete_threshold=3000, level_complete_reward=100.0):
    """
    Computes a bonus for reaching or surpassing a certain x position, which approximates the end of the level. 
    This encourages the agent to complete the level rather than just surviving or engaging in combat.
    Returns:
    - bonus (float): The computed bonus value for reaching the level completion threshold.
    Notes for Students:
    - This function can be integrated into the main reward function to provide a strong incentive for completing the level. Consider how to set the level completion threshold based on the specific level design and how to scale the reward to appropriately reflect the achievement of finishing the level.
    """
    try:
        vars_current_obs, vars_last_obs = vars_observation(current_obs, last_obs)
        if vars_current_obs['mario_pos'] is None:
            return 0
        
        cur_x, _ = vars_current_obs['mario_pos']

        # Large bonus for finishing the level (approximate end condition)
        if cur_x >= level_complete_threshold:
            return level_complete_reward
    except Exception:
        pass
    return 0