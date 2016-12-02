#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import gym
import random
from simpleMemory import Memory, RingBuffer
from keras.models import Sequential, model_from_config
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, sgd, Adam

ENV_NAME = 'Breakout-ram-v0'
#os.chdir('/home/edgard/Desktop/CS229-TetrisIsAwesome/CS-229 RL')
#os.chdir('/home/jennie/Desktop/CS229-TetrisIsAwesome/MaTris-master/')

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(436)
env.seed(416)
#env.reset()
mode='test'

################# FUNCTIONS ################
# Sets the frameskip and a Game Over signal to train and if testing, it plays the game normally.
def _step(a):
    reward = 0.0
    action = env._action_set[a]
    lives_before = env.ale.lives()
    for _ in range(4):
        reward += env.ale.act(action)
    ob = env._get_obs()
    done = env.ale.game_over() or (mode == 'train' and lives_before != env.ale.lives())
    #if done:
    #	print('DONE', "Action ", action)
    return ob, reward, done, {}

def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone
    
################# MODEL INITIALIZATION AND PARAMETERS ################
gamma = 0.99 # decay rate of past observations
warmup = 50000 # timesteps to observe before training
explore = 1000000 # frames over which to anneal epsilon
epsilon_tf = 0.1 # final value of epsilon
epsilon_t0 = 1 # starting value of epsilon
epsilon_test=0.005 #epsilon for testing purposes
memory_replay = 1000000 # number of previous transitions to remember
batch_size = 32 # size of minibatch
nb_steps = 50000000
train_visualize = False
resume=False
stepresume=960000
nodesperlayer=128

# Variables to set frameskip, target model, network
user_inputs = True
parser = argparse.ArgumentParser(description='ADD YOUR DESCRIPTION HERE')
parser.add_argument('-fs','--frameskip', help='Boolean for frameskip', default='T',
                    required=False)
parser.add_argument('-update','--update', help='Number of steps to update target', default=10000,
                    type = int, required=False)
parser.add_argument('-net','--linearNet', help='Boolean for linear network', default='F',
                    required=False)
args = parser.parse_args()
print(args)

frameskip = args.frameskip
update_target = args.update
linearNet = args.linearNet

#FRAME_PER_ACTION = 1
    
# Changing model structure
if frameskip == 'T':
    env._step = _step
nb_actions = env.action_space.n
state_size = env.observation_space.shape

# Initialize model 

if linearNet == 'T':
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
else: 
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(nodesperlayer,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(nodesperlayer,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(nodesperlayer,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

print(model.summary())

# Initialize target model
target_model = clone_model(model)
target_model.compile(Adam(lr=0.1), 'mse')

#model.compile(sgd(lr=0.2, clipvalue=1), 'mse')
#model.compile(Adam(lr=0.001, clipvalue=1), 'mse')
model.compile(Adam(lr=1e-6,clipvalue=1), 'mse')

################ TESTING ################
if mode == 'test':
    # Load model weights
    weights_filename = 'dqn_{0}_params_{1}_{2}_{3}.h5f'.format(
                    ENV_NAME, frameskip, update_target, linearNet)
    model.load_weights(weights_filename)

    # Testing model
    episodes = 5
    mode='test'
    for eps in range(1, episodes+1):
        # Start env monitoring
        exp_name = './Breakout-exp-' + str(eps) + '/'
        env.monitor.start(exp_name, force = True)
        env.reset()
        
        # Initialize outputs
        tReward = 0
        max_Q = 0
        terminal = False
        epsilon = epsilon_test
        
        # Initialize game with random action
        action_t0 = env.action_space.sample()
        state_t0, reward, terminal, info = env.step(action_t0)
        state_t = np.array(state_t0, dtype=float)/255
        state_t = state_t.reshape(1, 1, state_t0.shape[0])
        
        # Run the game until terminal
        while not terminal:
            # Select an action a
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                q = model.predict(state_t)
                max_Q = np.max(q)
                action = np.argmax(q)
            
            # Carry out action and observe new state state_t1 and reward
            state_t, reward, terminal, info = env.step(action)
            state_t = np.array(state_t, dtype=float)/255
            state_t = state_t.reshape(1, 1, state_t0.shape[0])
            tReward += reward
            #print("Eps", eps, "Reward ", tReward, "Max Q", max_Q)
        
        print("Eps", eps, "Reward ", tReward, "Max Q", max_Q)
        
        env.monitor.close()

    # TODO: plot average Q, score per episode
    #agent = Agent(model);
    #agent.train(env)
    #agent.test(env)

    #def frameStep(visualize, action):
    #    if visualize:
    #        env.render()
    #    state, reward, terminal, info = env.step(action)
    #    state_t = state.reshape(1, 1, state.shape[0])
    #    if terminal:
    #        env.reset()
    #    return [state_t, reward, terminal, info]

