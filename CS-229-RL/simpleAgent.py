#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:49:43 2016

@author: jiamingzeng
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os

class Agent:
    def __init__(self, model, memory, nb_step = 100000, warmup = 10000, batch_size = 32, 
                 epsilon_t0 = 0.1, epsilon_tf = 0.01, gamma = 0.99, explore = 300000):
        self.mem = memory
        self.model = model
        
        self.gamma = gamma # decay rate of past observations
        self.warmup = warmup # timesteps to observe before training
        self.explore = explore # frames over which to anneal epsilon
        self.epsilon_tf = epsilon_tf # final value of epsilon
        self.epsilon_t0 = epsilon_t0 # starting value of epsilon
        self.memory_replay = warmup # number of previous transitions to remember
        self.batch_size = batch_size # size of minibatch
        
    def train(self):
        
    def stepFrame(self):
        self.action_t0 = env.action_space.sample()
        env.render()
        state_t0, reward, terminal, info = env.step(action_t0)
        