import torch
import torch.nn as nn
import numpy as np
import marioai

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid() # Sigmoid for multi-binary output (or should it be separate?)
        )

    def forward(self, x):
        return self.model(x)

# Optimized MLPAgent with reduced input space and fixed enemy representation.
class MLPAgent(marioai.Agent):
    # Optimized input space: 7x7 window (49) + mario pos (2) + flags (2) + fixed enemy representation (6) = 59
    def __init__(self):
        super(MLPAgent, self).__init__()
        # Input dimension: 
        # 22x22 grid = 484
        # + 2 (mario pos) + 2 (can_jump, on_ground) + ?
        # Let's verify input size.
        # sense() gets: (mayMarioJump, isMarioOnGround, marioFloats, enemiesFloats, levelScene, dummy)
        # Flattened levelScene: 484
        # Mario floats: 2 (usually normalized or relative?)
        # Enemies floats: variable length... tricky for MLP.
        # For simplicity, let's stick to a fixed size input.
        # Reference `extractObservation` in utils.py tells us what we get.

        # Let's simplify inputs for the first version:
        # Flattened levelScene (22x22) = 484
        # marioFloats (x, y) = 2
        # isMarioOnGround = 1
        # mayMarioJump = 1
        # Total = 488

        # self.input_dim = 488
        
        # Mario only needs to look ahead and around him, not the entire level. 
        # Let's optimize input space. Reduced input space:
        # 7x7 window ahead/below Mario (rows 9-15, cols 11-17) = 49
        # 3 closest enemies as relative (dx, dy) = 6
        # marioFloats (x, y) = 2
        # isMarioOnGround = 1
        # mayMarioJump = 1
        # Total = 59
        
        self.input_dim = 59 # 49 (window) + 6 (enemies) + 2 (mario pos) + 2 (flags)
        self.output_dim = 5 # [backward, forward, crouch, jump, speed/bombs]
        
        self.mlp = MLP(self.input_dim, self.output_dim)

        # Action threshold
        self.threshold = 0.35 # Threshold for converting MLP outputs to binary actions

        # Consecutive jump management
        self.consecutive_jump_steps = 0
        self.max_consecutive_jump_steps = 5 # Max steps to keep jump active after initial activation

        # Wait time before allowing another jump activation (to prevent spamming jump)
        self.jump_cooldown = 0
        self.jump_cooldown_time = 5 # Cooldown time in steps after jump is deactivated before it can be activated again

        # Consecutive movement swap management (to prevent erratic switching between forward/backward/crouch)
        self.last_movement_action = None
        self.movement_swap_cooldown = 0
        self.movement_swap_cooldown_time = 10 # Cooldown time in steps after switching movement action before allowing another switch

    def sense(self, obs):
        super(MLPAgent, self).sense(obs)
        # obs is (mayMarioJump, isMarioOnGround, marioFloats, enemiesFloats, levelScene, dummy)
        # But wait, `Agent.sense` unpacks it.
        # self.can_jump
        # self.on_ground
        # self.mario_floats
        # self.enemies_floats
        # self.level_scene (numpy array 22x22)
        pass

    # Optimized window-based input processing and fixed enemy representation.
    def act(self):
        if self.level_scene is None:
            return [0, 0, 0, 0, 0] # No input yet, return no action

        # Flatten level scene
        # scene_flat = self.level_scene.flatten()

        # Mario position (keep as feature inputs)
        mario_pos = np.array(self.mario_floats)

        # Use the level_scene grid center for window extraction (Mario is centered at 11,11)
        cx, cy = 11, 11
        x_min = max(0, cx - 3)
        x_max = min(21, cx + 3)
        y_min = max(0, cy - 3)
        y_max = min(21, cy + 3)

        # Extract window and pad if necessary
        window = np.zeros((7, 7), dtype=self.level_scene.dtype)
        win_x_min = 3 - (cx - x_min)
        win_x_max = 3 + (x_max - cx) + 1
        win_y_min = 3 - (cy - y_min)
        win_y_max = 3 + (y_max - cy) + 1

        # Copy the valid part of the level_scene into the window
        window[win_y_min:win_y_max, win_x_min:win_x_max] = self.level_scene[y_min:y_max+1, x_min:x_max+1]
        scene_flat = window.flatten()  # 49 values

        # Boolean flags
        flags = np.array([float(self.can_jump), float(self.on_ground)])

        # Let's also include a fixed-size representation of the closest enemies. 
        # We will take the 3 closest enemies and represent them as relative positions (dx, dy) to Mario. 
        # If there are fewer than 3 enemies, we will pad with zeros.
        # 3 closest enemies as relative (dx, dy) — fixed-size enemy representation
        N_ENEMIES = 3
        enemy_features = np.zeros(N_ENEMIES * 2)
        if self.enemies_floats:
            enemies = sorted(
                self.enemies_floats,
                key=lambda e: (e[0] - mario_pos[0]) ** 2 + (e[1] - mario_pos[1]) ** 2
            )[:N_ENEMIES]
            for i, enemy in enumerate(enemies):
                ex, ey = enemy[0], enemy[1]  # 3-tuple (x, y, type)
                enemy_features[i * 2]     = ex - mario_pos[0]
                enemy_features[i * 2 + 1] = ey - mario_pos[1]

        # Concatenate inputs
        inputs = np.concatenate((scene_flat, mario_pos, flags, enemy_features))
        
        # Convert to tensor
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        # Forward pass
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        # Convert to action list
        action_probs = output_tensor.numpy() # Probabilities for each action
        action = (action_probs > self.threshold).astype(int).tolist() # Convert to binary actions based on threshold

        # print(f'Actions: {action}')

        # Avoid multiple movement actions at once (e.g., can't move forward and backward simultaneously)
        # Check which movement action (backward, forward, crouch) has the highest probability and set it to 1, others to 0
        movement_actions = action[:3] # First 3 are movement actions
        if sum(movement_actions) > 0: # If any movement action is active
            max_idx = np.argmax(movement_actions)
            # Set only the highest prob movement action to 1 but prioritize forward > backward > crouch in case of ties
            if movement_actions[max_idx] == movement_actions[1]: # If forward is tied for highest, prioritize it
                action[:3] = [0, 1, 0]
            elif movement_actions[max_idx] == movement_actions[0]: # If backward is tied for highest, prioritize it
                action[:3] = [1, 0, 0]
            else:
                action[:3] = [0, 0, 1] # Crouch if it's the highest or tied with lower priority

        # Check for movement swap and manage cooldown to prevent erratic switching
        current_movement_action = np.argmax(action[:3]) if sum(action[:3]) > 0 else None
        if current_movement_action != self.last_movement_action:
            if self.movement_swap_cooldown == 0:
                self.last_movement_action = current_movement_action
                self.movement_swap_cooldown = self.movement_swap_cooldown_time # Set cooldown after a movement switch
            else:
                # If we're in cooldown, revert to last movement action
                if self.last_movement_action is not None:
                    action[:3] = [1 if i == self.last_movement_action else 0 for i in range(3)]
                else:
                    action[:3] = [0, 0, 0] # No movement if last was None
                self.movement_swap_cooldown -= 1 # Decrease cooldown

        # Check if jump is active and manage consecutive jump steps to encourage keeping it active longer
        if action[3] == 1: # Jump action is active
            if self.jump_cooldown == 0: # Only allow jump if cooldown is 0
                self.consecutive_jump_steps += 1
                if self.consecutive_jump_steps > self.max_consecutive_jump_steps:
                    action[3] = 0 # Deactivate jump if we've exceeded max consecutive steps
                    self.jump_cooldown = self.jump_cooldown_time # Set cooldown
            else:
                action[3] = 0 # Deactivate jump if cooldown is active
                self.jump_cooldown -= 1 # Decrease cooldown
        else:
            self.consecutive_jump_steps = 0 # Reset counter if jump is not active

        return action

    def get_param_vector(self):
        params = []
        for param in self.mlp.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_param_vector(self, vector):
        offset = 0
        for param in self.mlp.parameters():
            shape = param.shape
            size = np.prod(shape)
            param.data = torch.tensor(vector[offset:offset + size].reshape(shape), dtype=torch.float)
            offset += size
