import numpy as np
import marioai
from enum import IntEnum

class Mario(IntEnum):
    KEY_LEFT = 0
    KEY_RIGHT = 1
    KEY_DOWN = 2
    KEY_JUMP = 3
    KEY_SPEED = 4

class Sprite(IntEnum):
    KIND_GOOMBA = 2
    KIND_GOOMBA_WINGED = 3
    KIND_RED_KOOPA = 4
    KIND_RED_KOOPA_WINGED = 5
    KIND_GREEN_KOOPA = 6
    KIND_GREEN_KOOPA_WINGED = 7
    KIND_BULLET_BILL = 8
    KIND_SPIKY = 9
    KIND_SPIKY_WINGED = 10

class CodeAgent(marioai.Agent):
    def __init__(self):
        super(CodeAgent, self).__init__()
        self._action_function = None
        self.code_str = ''
    
    @property
    def action_function(self):
        return self._action_function

    @action_function.setter
    def action_function(self, value):
        self.code_str = value
        code = compile(value, '<string>', 'exec')
        to_exec = {}
        exec(code, to_exec)
        self._action_function = to_exec['corre']

    def sense(self, obs):
        super(CodeAgent, self).sense(obs)
        
        # obs is (mayMarioJump, isMarioOnGround, marioFloats, enemiesFloats, levelScene, dummy)
        # But wait, `Agent.sense` unpacks it.
        # self.can_jump
        # self.on_ground
        # self.mario_floats
        # self.enemies_floats
        # self.level_scene (numpy array 22x22)
        pass

    def act(self):
        action = [0,0,0,0,0]
        if self.level_scene is not None:
            context = {'action': action, 
                        'Mario' : Mario,
                        'Sprite' : Sprite, 
                        'landscape' : self.level_scene, 
                        'enemies' : self.level_scene,
                        'can_jump' : self.can_jump,
                        'on_ground' : self.on_ground
                        }
            
            self._action_function(**context)
        return action

