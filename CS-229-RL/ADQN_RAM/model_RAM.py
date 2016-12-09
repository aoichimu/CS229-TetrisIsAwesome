import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions):
  with tf.device("/cpu:0"):
    state = tf.placeholder("float", [None, 128])
    inputs = Input(shape=(128,))
    model = Dense(256, activation='relu')(inputs)
    model =  Dense(256, activation='relu')(model)
    q_values = Dense(output_dim=num_actions, activation='linear')(model)
    m = Model(input=inputs, output=q_values)
  return state, m
