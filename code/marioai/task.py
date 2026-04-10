
import marioai

__all__ = ['Task']


class Task(object):
    '''A task handles communication with the environment.

    It decides how to evaluate the observations, potentially returning 
    reinforcement rewards or fitness values. Furthermore it is a filter for 
    what should be visible to the agent. Also, it can potentially act as a 
    filter on how actions are transmitted to the environment.

    Attributes:
      env (Environment): the environment instance.
      finished (bool): ?
      reward (int): the current reward of the simulation.
      status (int): ?
      cum_reward (int): the sum reward since the beginning of the episode.
      samples (int): number of steps in the current episode.
    '''


    def __init__(self, *args, **kwargs):
        '''Constructor.

        Args:
          environment (Environment): the environment instance.
        '''

        self.env = marioai.Environment(*args, **kwargs)
        self.finished = False
        self.reward = 0
        self.status = 0
        self.cum_reward = 0
        self.samples = 0
        self.last_observation = None

    def reset(self):
        '''Reinitialize the environment.'''

        self.env.reset()
        self.cum_reward = 0
        self.samples = 0
        self.finished = False
        self.reward = 0
        self.status = 0
        self.last_observation = None
    
    def enable_visualization(self):
        self.env.visualization = True
        #self.reset()

    @property
    def level_difficulty(self):
        return self.env.level_difficulty

    @level_difficulty.setter
    def level_difficulty(self, value):
        self.env.level_difficulty = value
        #self.reset()

    def get_sensors(self): 
        '''Bridge to environment.'''

        sense = self.env.get_sensors()
        # sense is now an Observation namedtuple
        
        if sense.level_scene is None:
            # Fitness packet (no level scene)
            self.reward = sense.distance # Or standard fitness handling
            self.status = sense.status
            self.finished = True
        else:
            # Step reward
            self.reward = self.compute_reward(sense, self.last_observation)
            self.last_observation = sense
            
        return sense

    def compute_reward(self, current_obs, last_obs):
        """
        Compute reward based on current and previous observations.
        You can override this or modify it to include more complex signals.
        """
        reward = 0
        
        return reward

    def perform_action(self, action):
        '''Bridge to environment.'''

        if not self.finished:
            self.env.perform_action(action)
            self.cum_reward += self.reward
            self.samples += 1


