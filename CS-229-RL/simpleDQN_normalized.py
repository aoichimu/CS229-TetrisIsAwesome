#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import gym
from simpleMemory import Memory, RingBuffer
from keras.models import Sequential, model_from_config
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, Adam
from keras import initializations

# Variables to set
parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('-env','--env', help='Game selection', default='Breakout-ram-v0',
                    required=False)
parser.add_argument('-net','--net', help='Network architecture', default='2',
                    required=False)
parser.add_argument('-opt','--opt', help='Optimizer', default='adam',
                    required=False)
parser.add_argument('-output','--output', help='Output folder', default='output',
                    required=False)
parser.add_argument('-mode','--mode', help='Mode', default='train',
                    required=False)
args = parser.parse_args()
print(args)

env = args.env
net = args.net
opt = args.opt
output = args.output
mode = args.mode

# Create directory for output
if not os.path.exists(output):
    os.makedirs(output)

ENV_NAME = env
#os.chdir('/home/edgard/Desktop/CS229-TetrisIsAwesome/CS-229 RL')
#os.chdir('/home/jennie/Desktop/CS229-TetrisIsAwesome/MaTris-master/')

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
env.reset()
#mode='train'

################# FUNCTIONS ################
# Sets the frameskip and a Game Over signal to train and if testing, it plays the game normally.
def _step(a):
    reward = 0.0
    action = env._action_set[a]
    lives_before = env.ale.lives()
    reward += env.ale.act(action)
    ob = env._get_obs()
    done = env.ale.game_over() #or (mode == 'train' and lives_before != env.ale.lives())
    if lives_before != env.ale.lives():
        done = True
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
warmup = 5000 # timesteps to observe before training
explore = 100000 # frames over which to anneal epsilon
epsilon_tf = 0.1 # final value of epsilon
epsilon_t0 = 1 # starting value of epsilon
epsilon_test=0.005 #epsilon for testing purposes
memory_replay = 100000 # number of previous transitions to remember
batch_size = 32 # size of minibatch
nb_steps = 5000000
train_visualize = False
saveweights=100
update_target = 1
frameskip = 'F'
nodesperlayer=128

# Changing model structure
if frameskip == 'T' and mode == 'train':
    print('Using framskip.')
env._step = _step
nb_actions = env.action_space.n
state_size = env.observation_space.shape

# Initialize model
if net == '2':
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(nodesperlayer*7))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
else:
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(nodesperlayer))
    model.add(Activation('relu'))
    model.add(Dense(nodesperlayer))
    model.add(Activation('relu'))
    model.add(Dense(nodesperlayer))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

print(model.summary())

# Initialize target model, random combilations
adam = Adam(lr=0.00025)
rmsprop = RMSprop(lr=0.00025, epsilon=0.1,rho = 0.95, decay=0.95)

target_model = clone_model(model)
target_model.compile(optimizer=adam,loss='mse')

if opt == 'rms':
    model.compile(optimizer=rmsprop,loss='mse')
else:
    model.compile(optimizer=adam,loss='mse')

################# TRAINING ################
if mode == 'train':
    # initialize action value function q
    action_t0 = env.action_space.sample()
    if train_visualize:
        env.render()
    state_t, reward, terminal, info = env.step(action_t0)
    state_t = np.float32(state_t / 255.0) # normalization of inputs
    state_t = state_t.reshape(1, 1, state_t.shape[0])

    # Start training
    epsilon = epsilon_t0
    t = 0
    eps = 0
    total_R = 0
    avg_Q = 0
    max_Q = 0
    
    memory = Memory(memorySize=memory_replay)
    # TODO: Implement Prioritized Experience Replay: 
    #    https://arxiv.org/pdf/1511.05952v4.pdf

    while t < nb_steps:
        # Initialize outputs
        loss = 0
        if t == warmup:
            eps = 1
        
        # Select an action a and save q value
        q = model.predict(state_t)
        #avg_maxQ += np.max(q)
        #avg_Q += np.mean(q)
        
        if np.random.uniform() <= epsilon:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q)
            
        
        # Carry out action and observe new state state_t1 and reward
        if train_visualize:
            env.render()
        state_t1, reward, terminal, info = env.step(action)
        state_t1 = np.float32(state_t1 / 255.0)
        state_t1 = state_t1.reshape(1, 1, state_t1.shape[0])
        
        if terminal:
            eps += 1 # increase episode count
            total_R += reward # update reward
            env.reset()
        
        # Linear anneal: We reduced the epsilon gradually
        if epsilon > epsilon_tf and t > warmup:
            epsilon -= (epsilon_t0 - epsilon_tf) / explore
        
        # Store experience
        memory.append(state_t, action, np.clip(reward, -1, 1), state_t1, terminal)
                
        # Sample random transitions from memory
        qInputs = np.zeros((batch_size, state_t.shape[1], state_t.shape[2]))
        targets = np.zeros((batch_size, nb_actions))
        
        if t > warmup:
            minibatch = memory.randSample(batch_size)
        
            for i in range(0, len(minibatch)):
                ss, aa, rr, ss_t1, terminal = minibatch[i]
                #print(minibatch[i])
                qInputs[i:i+1] = ss
                targets[i] = model.predict(ss)

                if terminal:
                    targets[i, aa] = rr
                else:
                    qTarget = target_model.predict(ss_t1)
                    max_Q = np.max(qTarget)
                    avg_Q = np.mean(qTarget)
                    tt = rr + gamma*max_Q
                
                    targets[i, aa] = tt
            loss += model.train_on_batch(qInputs, targets)
            
        # Update target model
        if (t % update_target == 0):
            target_model.set_weights(model.get_weights())
        
        t += 1
        #trainTime += 1
        state_t = state_t1
        
        # Save weights and output periodically
        if (eps % saveweights == 0 and t > warmup):
            print("Time", t, "Eps", eps,
                  #"Train time", trainTime,
                  "Loss ", '%.2E' % loss,
                  "Max Q", '%.2E' % max_Q,
                  "Avg Q", '%.2E' % avg_Q,
                  "Total R", total_R)
            total_R = 0
            #trainTime = 0
            
            model.save_weights('{3}/dqn_{0}_RAM_{1}_{2}.h5f'.format(
                ENV_NAME, opt, t, output), overwrite=True)

# Close files that were written
#all_loss.close()
#all_Q.close()        
############# PLOTTING ################

################ TESTING ################
if mode == 'test':
    # Load model weights
    weights_filename = '{3}/dqn_{0}_RAM_{1}_{2}.h5f'.format(
                ENV_NAME, opt, t, output)
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
            if no.random.uniform() <= epsilon:
                action = np.random.random_integers(0, nb_actions-1)
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
