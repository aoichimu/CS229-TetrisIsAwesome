from Atari_Preprocess import AtariEnvironment
import threading
import time
import tensorflow as tf

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

from keras.models import Sequential, model_from_config
from keras.initializations import normal, identity
from keras.layers import Dense, Activation, Flatten, Permute
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, sgd, Adam
from keras import initializations

######USER INPUTS#######
parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('-env','--env', help='Game selection', default='Breakout-v0',
                    required=False)
parser.add_argument('-init','--init', help='Initialization', default='glorot_uniform',
                    required=False)
parser.add_argument('-opt','--opt', help='Optimizer', default='rms',
                    required=False)
parser.add_argument('-threads','--threads', help='Threads', default='1',
                    type = int, required=False)
parser.add_argument('-output','--output', help='Output folder', default='output',
                    required=False)
parser.add_argument('-mode','--mode', help='Mode', default='train',
                    required=False)
args = parser.parse_args()
print(args)

env = args.env
initDist = args.init
opt = args.opt
mode = args.mode
threads = args.threads
output = args.output

# Create directory for output
if not os.path.exists(output):
    os.makedirs(output)

######VARIABLES######
ENV_NAME = env
gamma = 0.99 # decay rate of past observations
explore = 1000000 # frames over which to anneal epsilon
TMAX = 50000000
printQ=5000
UPDATE_NETWORK=5
UPDATE_TARGET_NETWORK=40000
T=0
IMG_WIDTH=80
IMG_HEIGHT=80
FRAME_PER_ACTION=4
img_channels=4
saved=False
count = 0
#####################
def sample_final_epsilon():
    #Samples a final epsilon, based on the asynchronous model of Minh
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def create_model(nb_actions,img_rows,img_cols,opt):
#Creates a model like the Minh one

    INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
    WINDOW_LENGTH = 4

    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',init=initDist))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same',init=initDist))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same',init=initDist))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512,init=initDist))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions,init=initDist))
    model.add(Activation('linear'))
    print(model.summary())

    if opt == 'rms':
        model.compile(RMSprop(lr=0.00025, epsilon=0.1, rho = 0.95,
                              decay=0.95, clipvalue=1), 'mse')
    else:
        model.compile(Adam(lr=0.00025, clipvalue=1), 'mse')
        
    return model
    
def clone_model(model, custom_objects={}):
    #Makes an identical copy of an existing model
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    clone.compile(RMSprop(lr=0.00025, epsilon=0.1, rho = 0.95,
                          decay=0.95, clipvalue=1), 'mse') 
    return clone

def actor_learner_thread(thread_id,OLD_ENV, model,target_model,num_actions):
    #One thread, global variables
    global TMAX, T, saved, count
    global graph
    with graph.as_default():
    
        #Wrap the environment so we can safely use it
        env=AtariEnvironment(gym_env=OLD_ENV, resized_width=IMG_WIDTH, resized_height=IMG_HEIGHT)
        epsilon_tf = sample_final_epsilon()
        epsilon_t0 = 1
        epsilon=epsilon_t0
        
        
        #Wait, so there is no conflicts
        time.sleep(3*thread_id)
        print("Starting thread ", thread_id, "with final epsilon ", epsilon_tf)
        #START the thread
        t = 0
        i=0
        while T < TMAX:
            #Start the environment
            state_t = env.get_initial_state()
            inputs_batch = np.zeros((UPDATE_NETWORK, state_t.shape[1],state_t.shape[2], state_t.shape[3]))
            target_batch = np.zeros((UPDATE_NETWORK, num_actions))
            terminal=False
            TReward=0
            survived=0
            avgQ=0
            while True: #this is while not terminal
                # Select an action a
                q = model.predict(state_t)
                target_batch[i]=q
                "selected an action"
                if t%FRAME_PER_ACTION ==0:
                    if random.random() <= epsilon:
                         action = env.env.action_space.sample()
                    else:
                         action = np.argmax(q)
                # Anneal epsilon
                if epsilon > epsilon_tf:
                    epsilon -= (epsilon_t0 - epsilon_tf) / explore
                
                #Observe next state and fill minibatch
                state_t1, reward, terminal, info = env.step(action)
                target=reward
                if not terminal:
                    target_q=target_model.predict(state_t1)
                    target+=gamma*np.max(target_q)
                    
                inputs_batch[i]=state_t
                target_batch[i,action]=target
                #UPDATE COUNTERS AND TRANSITION
                state_t = state_t1
                T += 1
                t += 1
                i+=1
                survived+=1
                avgQ+=np.max(q)
                TReward+=reward
                if T%UPDATE_TARGET_NETWORK==0:
                    count += 1
                    target_model.set_weights(model.get_weights())
                    print("Thread ", thread_id, "updated the target model ", count*UPDATE_TARGET_NETWORK)
                    target_model.save_weights('{4}/dqn_Asynchronous_{0}_{1}_{2}_{3}.h5f'.format(
                        ENV_NAME, initDist, opt, count*UPDATE_TARGET_NETWORK, output), overwrite=True)
                   
                if t%UPDATE_NETWORK==0:
                    i=0
                    loss = model.train_on_batch(inputs_batch,target_batch) 
                if terminal:
                    loss = model.train_on_batch(
                        inputs_batch[:i],target_batch[:i,:])
                    print("Thread# ", thread_id,"EpsReward: ",TReward, "avgQ: ", avgQ/survived,"Time: ",T)
                    break
                    
############################################################################
#MAIN CODE HERE
num_threads=threads
envs = [gym.make(ENV_NAME) for i in range(num_threads)]
model=create_model(envs[0].action_space.n,IMG_WIDTH,IMG_HEIGHT,opt)
target_model=clone_model(model)
graph = tf.get_default_graph()
actor_learner_threads = [threading.Thread(target=actor_learner_thread,args=(thread_id, envs[thread_id],model,target_model,envs[0].action_space.n)) for thread_id in range(num_threads)]
target_model.save_weights('TEST/dqn_Asynchronous_0.h5f', overwrite=True)
for tau in actor_learner_threads:
    tau.start()
       
