import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque

class AtariEnvironment(object):
    """
    Wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to the latest state
    the agent has experienced. Its history has 4 frames at the moment
    """
    def __init__(self, gym_env, resized_width, resized_height):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = 4

        self.gym_actions = gym_env.action_space.n
        
        # Our memory is going to be a deque one for now, we will use it to
        #construct the states
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer
        Apparently it helps resize the arrays automatically
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.preprocess(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 0)
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)
        return s_t

    def preprocess(self, observation):
        """
        Converts the image to grayscale and resizes it
        """
        return resize(rgb2gray(observation), (self.resized_width, self.resized_height))

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        lives_before = self.env.ale.lives()
    
        x_t1, r_t, terminal, info = self.env.step(action_index)
        if lives_before != self.env.ale.lives():
            terminal=1
            
        x_t1 = self.preprocess(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        s_t1 = s_t1.reshape(1, s_t1.shape[0], s_t1.shape[1], s_t1.shape[2])
        return s_t1, r_t, terminal, info
