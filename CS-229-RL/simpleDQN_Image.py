#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import gym
import random

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from simpleMemory import Memory, RingBuffer
from keras.models import Sequential, model_from_config
from keras.initializations import normal, identity
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, sgd, Adam
from keras import initializations

ENV_NAME = 'Breakout-v0'
#os.chdir('/home/edgard/Desktop/CS229-TetrisIsAwesome/CS-229 RL')
#os.chdir('/home/jennie/Desktop/CS229-TetrisIsAwesome/MaTris-master/')

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
env.reset()
mode='train'

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
warmup = 10000 # timesteps to observe before training
explore = 50000 # frames over which to anneal epsilon
epsilon_tf = 0.1 # final value of epsilon
epsilon_t0 = 1 # starting value of epsilon
epsilon_test=0.005 #epsilon for testing purposes
memory_replay = 50000 # number of previous transitions to remember
batch_size = 32 # size of minibatch
nb_steps = 5000000
train_visualize = False
saveweights=5

resume=False
stepresume=960000
nodesperlayer=128


img_rows , img_cols = 80, 80
img_channels = 4 #We stack 4 frames , much originality, such coding skills, wow

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

print("Frameskip: ", frameskip, "Update Target: ", update_target,
      "Linear Net: ", linearNet)

FRAME_PER_ACTION = 1
# Changing model structure
if frameskip == 'T' and mode == 'train':
    print('Using framskip.')
    FRAME_PER_ACTION= 4
nb_actions = env.action_space.n
state_size = env.observation_space.shape

# Initialize model 
model = Sequential()
model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(
    shape, scale=0.01, name=name),border_mode='same',input_shape= (img_channels,img_rows,img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
model.add(Activation('relu'))
model.add(Dense(2,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

print(model.summary())

# Initialize target model
#target_model = clone_model(model)
#target_model.compile(RMSprop(lr=0.00025, epsilon=0.1,
#                             rho = 0.95, decay=0.95, clipvalue=1), 'mse')

#model.compile(sgd(lr=0.2, clipvalue=1), 'mse')
#model.compile(Adam(lr=0.001, clipvalue=1), 'mse')
model.compile(RMSprop(lr=0.00025, epsilon=0.1,
                      rho = 0.95, decay=0.95, clipvalue=1), 'mse')

if resume:
    print("Resuming training \n")
    weights_filename = 'dqn_{}_paramsRMSN.h5f'.format(ENV_NAME)
    model.load_weights(weights_filename)
    target_model.load_weights(weights_filename)
    epsilon = epsilon_t0-(epsilon_tf-epsilon_t0)*stepresume/explore

################# TRAINING ################
if mode == 'train':
    # initialize action value function q
    action_t0 = env.action_space.sample()
    if train_visualize:
        env.render()
    #initialize the frames x and stack 'img_channels' of them 
    x_t_colored, reward, terminal, info = env.step(action_t0)
    
    x_t = skimage.color.rgb2gray(x_t_colored)
    x_t = skimage.transform.resize(x_t,(img_rows,img_cols))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    state_t= np.stack((x_t, x_t, x_t, x_t), axis=0)
    state_t = state_t.reshape(1, state_t.shape[0], state_t.shape[1], state_t.shape[2])
    # Start training
    epsilon = epsilon_t0

    t = 0
    # Basic Deque memory (should upgrade later)
    memory = Memory(memorySize=memory_replay)
    # TODO: Implement Prioritized Experience Replay: 
    #    https://arxiv.org/pdf/1511.05952v4.pdf

    # TODO: Future Agent class
    #all_Q = open("maxQ.txt", "w")
    #all_loss = open("loss.txt", "w")

    while t < nb_steps:
        # Initialize outputs
        loss = 0
        rr = 0
        action = 0
        max_Q = 0
        avg_Q = 0
        
        # Select an action a
        if t%FRAME_PER_ACTION ==0:
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                q = model.predict(state_t)
                action = np.argmax(q)
        
        # Carry out action and observe new state state_t1 and reward
        if train_visualize:
            env.render()
        x_t_colored, reward, terminal, info = env.step(action)
        x_t = skimage.color.rgb2gray(x_t_colored)
        x_t = skimage.transform.resize(x_t,(img_rows,img_cols))
        x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
        x_t = x_t.reshape(1, 1, x_t.shape[0], x_t.shape[1])
        state_t1=np.append(x_t, state_t[:, :(img_channels-1), :, :], axis=1)
        
        if terminal:
            env.reset()
        
        # Linear anneal: We reduced the epsilon gradually
        if epsilon > epsilon_tf and t > warmup:
            epsilon -= (epsilon_t0 - epsilon_tf) / explore
        
        # Store experience
        memory.append(state_t, action, np.clip(reward, -1, 1), state_t1, terminal)
        
        if t > warmup:
            # Sample random transitions from memory
            qInputs = np.zeros((batch_size, state_t.shape[1], state_t.shape[2], state_t.shape[3]))
            targets = np.zeros((batch_size, nb_actions))
            minibatch = memory.randSample(batch_size)
        
            for i in range(0, len(minibatch)):
                ss, aa, rr, ss_t1, terminal = minibatch[i]
                targets[i] = model.predict(ss)
                #max_Q2=np.max(targets[i])
                qInputs[i:i+1] = ss

                if terminal:
                    targets[i, aa] = rr
                else:
                    #qTarget = target_model.predict(ss_t1)
                    qTarget = model.predict(ss_t1)
                    max_Q = np.max(qTarget)
                    avg_Q = np.mean(qTarget)
                    #print("Max_Q updated t=",t)
                    tt = rr + gamma*max_Q
                
                    targets[i, aa] = tt
        
            loss += model.train_on_batch(qInputs, targets)
            #all_loss.write('\n' + str(loss))
            #all_Q.write(str(max_Q)+ '\t' + str(max_Q2)+'\n')
            
        # Update target model
        #if (t % update_target == 0):
        #    target_model.set_weights(model.get_weights())
        
        t += 1
        state_t = state_t1
        
        # Save weights and output periodically
        if (t % saveweights == 0):
            print("Time", t, "Loss ", '%.2E' % loss, "Max Q", max_Q,
                  "Avg Q", avg_Q, "Action ", action)
            model.save_weights('161204_exp2/dqn_{0}_paramsRMS_IMG_{1}_{2}_{3}_{4}.h5f'.format(
                ENV_NAME, frameskip, update_target, linearNet, t), overwrite=True)

# Close files that were written
#all_loss.close()
#all_Q.close()        
############# PLOTTING ################

################ TESTING ################
if mode == 'test':
    # Load model weights
    weights_filename = 'dqn_{0}_paramsRMS_{1}_{2}_{3}.h5f'.format(
                    ENV_NAME, frameskip, update_target, linearNet)
    model.load_weights(weights_filename)

    # Testing model
    episodes = 5
    for eps in range(1, episodes+1):
        # Start env monitoring
        np.random.seed()
        env.seed()
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
        state_t = state_t0
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
            state_t = state_t.reshape(1, 1, state_t0.shape[0])
            tReward += reward
        
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
