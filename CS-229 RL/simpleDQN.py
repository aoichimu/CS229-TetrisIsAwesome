#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:44:22 2016

@author: jiamingzeng
"""

import numpy as np
import gym
from collections import deque
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import sgd

ENV_NAME = 'Breakout-ram-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
state_size = env.observation_space.shape

# Setting values
gamma = 0.99 # decay rate of past observations
warmup = 300 # timesteps to observe before training
#EXPLORE = 3000000. # frames over which to anneal epsilon
epsilon_tf = 0.0001 # final value of epsilon
epsilon_t0 = 0.1 # starting value of epsilon
memory_replay = 50000 # number of previous transitions to remember
batch_size = 32 # size of minibatch
#FRAME_PER_ACTION = 1

# Initialize model 

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

#S = Input(shape=state_size)
#h = Convolution2D(16, 8, 8, subsample=(4, 4),
#    border_mode='same', activation='relu')(S)
#h = Convolution2D(32, 4, 4, subsample=(2, 2),
#    border_mode='same', activation='relu')(h)
#h = Flatten()(h)
#h = Dense(256, activation='relu')(h)
#V = Dense(nb_actions)(h)
#model = Model(S, V)

print(model.summary())

model.compile(sgd(lr=0.2,clipvalue=1), 'mse')

# Basic Deque memory (should upgrade later)
memory = deque()
# TODO: Implement Prioritized Experience Replay: 
#    https://arxiv.org/pdf/1511.05952v4.pdf

# TODO: Future Agent class

# output of the model
# initialize action value function q
#action_initial = np.zeros([nb_actions])
#initial[0] = 1
action_t0 = env.action_space.sample()
state_t0, reward, terminal, info = env.step(action_t0)

epsilon = epsilon_t0
state_t = state_t0
#action_t = action_t0
t = 0
while t < 5000:
    loss = 0
    # Reshape state_t
    state_t = state_t.reshape(1, 1, state_t0.shape[0])
    
    # Select an action a
    if random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        q = model.predict(state_t)
        max_Q = np.argmax(q)
        action = max_Q
    
    # Carry out action and observe new state state_t1 and reward
    state_t1, reward, terminal, info = env.step(action)
    state_t1 = state_t1.reshape(1, 1, state_t0.shape[0])
    
    # TODO: check reward clipping to -1, 0, 1
    # TODO: Add linear anneal for epsilon greedy
    
    # Store experience
    memory.append((state_t, action, reward, state_t1, terminal))
    if len(memory) > memory_replay:
            memory.popleft()
            
    # Sample random transitions from memory
    qInputs = np.zeros((batch_size, state_t.shape[1], state_t.shape[2]))
    targets = np.zeros((batch_size, nb_actions))
    
    if t > warmup:
        minibatch = random.sample(memory, batch_size)
    
        for i in range(0, len(minibatch)):
            ss, aa, rr, ss_t1, terminal = minibatch[i]
            targets[i] = model.predict(state_t)
            qInputs[i:i+1] = ss

            if terminal:
                targets[i, aa] = rr
            else:
                qTarget = model.predict(ss_t1)
                max_Q = np.max(qTarget)
                tt = rr + gamma*max_Q
            
                targets[i, aa] = tt
    
    # TODO: clip delta of tt - Q(ss,aa) between 1 and -1, rewrite train_on_batch
    # DONE: Changed the specification of sgd on the model to clip the values
        loss += model.train_on_batch(qInputs, targets)
    
    print("TIMESTEP", t, "/ Loss ", '%.2E' % loss)
    # TODO: quit it when it converges
    t += 1

# Testing model
model.save_weights('dqn_{}_params.h5f'.format(ENV_NAME), overwrite=True)
model.tes

# TODO: plot average Q, score per episode
#agent = Agent(model);
#agent.train(env)
#agent.test(env)
