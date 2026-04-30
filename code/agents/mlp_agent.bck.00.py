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

class MLPAgent(marioai.Agent):
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
        
        self.input_dim = 488
        self.output_dim = 5 # [backward, forward, crouch, jump, speed/bombs]
        
        self.mlp = MLP(self.input_dim, self.output_dim)
        
        # Action threshold
        self.threshold = 0.5

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

    def act(self):
        if self.level_scene is None:
            return [0, 0, 0, 0, 0]

        # Flatten level scene
        scene_flat = self.level_scene.flatten()
        
        # Mario position (we might want to normalize this or just inputs, but let's take raw for now)
        mario_pos = np.array(self.mario_floats)
        
        
        # Boolean flags
        flags = np.array([float(self.can_jump), float(self.on_ground)])
        
        
        # Concatenate inputs
        # Note: enemies_floats turned out to be variable, ignoring for now or we need a fixed representation (e.g. closest enemy)
        inputs = np.concatenate((scene_flat, mario_pos, flags))
        
        # Convert to tensor
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        # Forward pass
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)
            
        # Convert to action list
        action_probs = output_tensor.numpy()
        action = (action_probs > self.threshold).astype(int).tolist()
        
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
