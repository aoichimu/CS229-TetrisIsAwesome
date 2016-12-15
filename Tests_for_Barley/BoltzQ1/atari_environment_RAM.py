import tensorflow as tf
import numpy as np
from collections import deque

class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer 
    of size agent_history_length from which environment state
    is constructed.
    """
    def __init__(self, gym_env):
        self.env = gym_env

        self.gym_actions = range(gym_env.action_space.n)
        if (gym_env.spec.id == "Pong-ram-v0" or gym_env.spec.id == "Breakout-ram-v0"):
            print "Doing workaround for pong or breakout"
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self.gym_actions = [1,2,3]

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, width, height]
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        s_t = self.get_preprocessed_RAM(x_t)
        
        
        return s_t

    def get_preprocessed_RAM(self, observation):
        """
        See Methods->Preprocessing in the Atari-RAM paper
        1) Normalize the input to lie between 0 and 1
        """
        return np.float32(observation / 255.0)

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        lives_before=self.env.ale.lives()
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        s_t1 = self.get_preprocessed_RAM(x_t1)
        if lives_before != self.env.ale.lives():
            terminal=True               
        
        return s_t1, r_t, terminal, info
