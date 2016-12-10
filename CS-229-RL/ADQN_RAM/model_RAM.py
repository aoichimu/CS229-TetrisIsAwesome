import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions, network):
  if network == 1:
    with tf.device("/cpu:0"):
      state = tf.placeholder("float", [None, 128])
      inputs = Input(shape=(128,))
      model = Dense(256, activation='relu')(inputs)
      model =  Dense(256, activation='relu')(model)
      q_values = Dense(output_dim=num_actions, activation='linear')(model)
      m = Model(input=inputs, output=q_values)
    return state, m
  elif network == 2:
    with tf.device("/cpu:0"):
      state = tf.placeholder("float", [None, 128])
      inputs = Input(shape=(128,))
      model = Dense(128, activation='relu')(inputs)
      model =  Dense(128, activation='relu')(model)
      q_values = Dense(output_dim=num_actions, activation='linear')(model)
      m = Model(input=inputs, output=q_values)
    return state, m
  elif network == 3:
    with tf.device("/cpu:0"):
      state = tf.placeholder("float", [None, 128])
      inputs = Input(shape=(128,))
      model = Dense(256, activation='relu')(inputs)
      model =  Dense(256, activation='relu')(model)
      model =  Dense(256, activation='relu')(model)
      q_values = Dense(output_dim=num_actions, activation='linear')(model)
      m = Model(input=inputs, output=q_values)
    return state, m
  elif network == 4:
    with tf.device("/cpu:0"):
      state = tf.placeholder("float", [None, 128])
      inputs = Input(shape=(128,))
      model = Dense(128, activation='relu')(inputs)
      model =  Dense(128, activation='relu')(model)
      model =  Dense(128, activation='relu')(model)
      q_values = Dense(output_dim=num_actions, activation='linear')(model)
      m = Model(input=inputs, output=q_values)
    return state, m
  elif network == 5:
    with tf.device("/cpu:0"):
      state = tf.placeholder("float", [None, 128])
      inputs = Input(shape=(128,))
      model = Dense(256, activation='relu')(inputs)
      model =  Dense(256, activation='relu')(model)
      model =  Dense(256, activation='relu')(model)
      model =  Dense(256, activation='relu')(model)
      q_values = Dense(output_dim=num_actions, activation='linear')(model)
      m = Model(input=inputs, output=q_values)
    return state, m
  else:
    with tf.device("/cpu:0"):
      state = tf.placeholder("float", [None, 128])
      inputs = Input(shape=(128,))
      model = Dense(128, activation='relu')(inputs)
      model =  Dense(128, activation='relu')(model)
      model =  Dense(128, activation='relu')(model)
      model =  Dense(128, activation='relu')(model)
      q_values = Dense(output_dim=num_actions, activation='linear')(model)
      m = Model(input=inputs, output=q_values)
    return state, m
