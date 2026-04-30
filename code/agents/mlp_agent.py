import torch
import torch.nn as nn
import numpy as np
import marioai


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
            # No Sigmoid: threshold raw logits at 0.0 for crisper binary decisions
        )


    def forward(self, x):
        return self.model(x)


# MLPAgent with compact input space and small network for efficient GA search.
class MLPAgent(marioai.Agent):
    # Input space: 7x7 forward window (49) + mario pos (2) + flags (2) + 3 closest enemies (dx,dy) (6) = 59
    # Network: 59 → 16 → 8 → 5 = 1085 parameters
    def __init__(self):
        super(MLPAgent, self).__init__()
        # Input space (all normalized to [-1, 1]):
        # 7x7 window: Mario at leftmost col, 6 cols ahead, 3 rows above/below (rows 8:15, cols 11:18) = 49
        # mario_pos: world (x, y) normalized by level dimensions = 2
        # flags: can_jump, on_ground mapped to {-1, 1} = 2
        # 3 closest enemies as relative (dx, dy) offsets, normalized by max range = 6
        # Total = 59

        self.input_dim = 59
        self.output_dim = 5  # [backward, forward, crouch, jump, speed/fire]

        self.mlp = MLP(self.input_dim, self.output_dim)


    def sense(self, obs):
        super(MLPAgent, self).sense(obs)
        # Populates: self.can_jump, self.on_ground, self.mario_floats,
        #            self.enemies_floats, self.level_scene
        pass


    def act(self):
        if self.level_scene is None:
            return [0, 0, 0, 0, 0] # No input yet, return no action

        full_window = self.level_scene # 22x22 grid

        # Forward-only 7x7 window: Mario at leftmost column, 6 cols ahead.
        # Mario is fixed at (col=11, row=11) in the 22x22 grid, so all indices
        # are always in-bounds — no padding required.
        #   cols: 11 (Mario) … 17 (6 ahead)
        #   rows:  8 (3 above) … 14 (3 below)
        window = full_window[8:15, 11:18]   # shape (7, 7)
        scene_flat = (window.flatten() - 15.5) / 26.5  # center+scale tile values [-11,42] → [-1,1]

        # Mario position normalized to [-1, 1]
        mario_pos = np.array([(self.mario_floats[0] - 1200.0) / 1200.0, (self.mario_floats[1] - 127.5) / 127.5])

        # Boolean flags mapped to {-1, 1}
        flags = np.array([2.0 * float(self.can_jump) - 1.0, 2.0 * float(self.on_ground) - 1.0])

        # 3 closest enemies as (dx, dy) relative to Mario, normalized to [-1, 1]
        N_ENEMIES = 3
        enemy_features = np.zeros(N_ENEMIES * 2)  # zero-padded if fewer than 3 enemies
        if self.enemies_floats:
            enemies = sorted(
                self.enemies_floats,
                key=lambda e: e[0]**2 + e[1]**2  # enemies are already (dx, dy) relative to Mario
            )[:N_ENEMIES]
            for i, enemy in enumerate(enemies):
                ex, ey = enemy[0], enemy[1]  # already (dx, dy) relative offsets from Mario
                enemy_features[i * 2]     = ex / 256.0  # scale [-256,256] → [-1,1]
                enemy_features[i * 2 + 1] = ey / 256.0  # scale [-256,256] → [-1,1]

        # Concatenate all inputs into a single feature vector
        inputs = np.concatenate((scene_flat, mario_pos, flags, enemy_features))
        
        # Convert to tensor
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        # Forward pass
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        # Threshold raw logits at 0.0 for binary actions
        action = (output_tensor.numpy() > 0.0).astype(int).tolist()

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
