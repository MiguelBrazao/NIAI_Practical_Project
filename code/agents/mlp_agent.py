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
        
        # Optimized input space:
        # 7x7 window ahead/below Mario (rows 9-15, cols 11-17) = 49
        # marioFloats (x, y) = 2
        # flags (isMarioOnGround, mayMarioJump) = 2
        # 3 closest enemies as relative (dx, dy) = 6
        # Bricks and items are NOT added separately: the 7x7 window already encodes
        # them within the jump_threshold range, so adding them again would only
        # increase the GA search space (~500 extra weights) for zero information gain.
        # Total = 49 + 2 + 2 + 6 = 59
        
        self.input_dim = 59 # 49 (window) + 2 (mario pos) + 2 (flags) + 6 (enemies)
        self.output_dim = 5 # [backward, forward, crouch, jump, speed/bombs]
        
        self.mlp = MLP(self.input_dim, self.output_dim)

        # Action threshold
        self.threshold = 0.25 # Threshold for converting MLP outputs to binary actions


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
    # Action post-processing to manage movement and jump behavior more effectively.
    def act(self):
        if self.level_scene is None:
            return [0, 0, 0, 0, 0] # No input yet, return no action

        full_window = self.level_scene # 22x22 grid

        # Use the full_window grid center for window extraction (Mario is centered at 11,11)
        cx, cy = 11, 11
        x_min = max(0, cx - 3)
        x_max = min(21, cx + 3)
        y_min = max(0, cy - 3)
        y_max = min(21, cy + 3)

        # Extract window and pad if necessary
        window = np.zeros((7, 7), dtype=full_window.dtype)
        win_x_min = 3 - (cx - x_min)
        win_x_max = 3 + (x_max - cx) + 1
        win_y_min = 3 - (cy - y_min)
        win_y_max = 3 + (y_max - cy) + 1

        # Copy the valid part of the full_window into the window
        window[win_y_min:win_y_max, win_x_min:win_x_max] = full_window[y_min:y_max+1, x_min:x_max+1]
        scene_flat = window.flatten()  # 49 values

        # Mario position (keep as feature inputs)
        mario_pos = np.array(self.mario_floats)

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

        # Extract coins and power-ups as additional features if needed (not implemented here, but could be added similarly to enemies)
        # Concatenate inputs (no brick/item features — redundant with scene_flat)
        inputs = np.concatenate((scene_flat, mario_pos, flags, enemy_features))
        
        # Convert to tensor
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        # Forward pass
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        # Convert to action list
        action_probs = output_tensor.numpy() # Probabilities for each action
        action = (action_probs > self.threshold).astype(int).tolist() # Convert to binary actions based on threshold

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
