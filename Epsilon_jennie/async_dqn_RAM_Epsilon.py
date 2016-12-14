#!/usr/bin/env python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from atari_environment_RAM import AtariEnvironment
import threading
import tensorflow as tf
import sys
import random
import numpy as np
import time
import gym
import signal
from keras import backend as K
from model_RAM import build_network


flags = tf.app.flags

flags.DEFINE_string('experiment', 'dqn_breakout_RAM', 'Name of the current experiment')
flags.DEFINE_string('game', 'Breakout-ram-v0', 'Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 80000000, 'Number of training timesteps.')
flags.DEFINE_integer('agent_history_length', 4, 'Use this number of recent screens as the environment state.')
flags.DEFINE_integer('network_update_frequency', 32, 'Frequency with which each actor learner thread does an async gradient update')
flags.DEFINE_integer('target_network_update_frequency', 10000, 'Reset the target network every n timesteps')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_float('epsilon_test', 0.005, 'epsilon for testing.')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, 'Number of timesteps to anneal temperatures.')
flags.DEFINE_string('summary_dir', './summaries', 'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Directory for storing model checkpoints')
flags.DEFINE_integer('summary_interval', 5,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval.')
flags.DEFINE_integer('checkpoint_interval', 100000,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds (rounded up to statistics interval.')
flags.DEFINE_boolean('show_training', False, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', '/media/edgard/ExtraDrive1/ADQN_RAM/Q-Boltz/checkpoints/Breakout_Boltzmann.ckpt-8550600', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_string('eval_dir', './evals', 'Directory to store gym evaluation')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')
flags.DEFINE_boolean('cont',False, 'Continue the training after annealing is done')
FLAGS = flags.FLAGS
T = 0
TMAX = FLAGS.tmax

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]
    
def choose_final_epsilon(thread_id):
    """
    Chooses a final epsilon for an 8-threaded experiment
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.5,.5,.5,.1,.1,.1,.01,.01])
    return final_epsilons[thread_id]
    
def temperature(epsilon,f,num_actions):
    """
    Sample a final temperature for a wild card Boltzmann Q Policy
    
    """
    
    return np.log(num_actions*(1.0/epsilon-1.0)+1.0)/f

def actor_learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX, T
        
    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env)
   
    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]
    cost = graph_ops["cost"]
    summary_placeholders, update_ops, summary_op = summary_ops
    
    # Initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []
    
    #Setup Summaries for the writer thread
    if thread_id==FLAGS.num_concurrent/4:
        summary_op = summary_ops[-1]
        summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
        writer = tf.train.SummaryWriter(summary_save_path, session.graph)
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        last_summary_time=0
    #Initialize epsilon for both continuing and not continuing
    time.sleep(3*thread_id)
    final_epsilon = choose_final_epsilon(thread_id)
    initial_epsilon = 1.0
    if not FLAGS.cont:
        epsilon = initial_epsilon
        print "Starting thread ", thread_id, "with final epsilon ", final_epsilon
    else:
        epsilon=final_epsilon
        print "Continuing thread ", thread_id, "with final epsilon ", final_epsilon
    t = 0
    while T < TMAX:
    
    
    
        # Get initial game observation
        s_t = env.get_initial_state()
        terminal = False

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0
        total_loss=0
        
        while True:
        
            # Check for summary writing
            if thread_id==FLAGS.num_concurrent/4:
                now = time.time()
                if now - last_summary_time > FLAGS.summary_interval:
                    summary_str = session.run(summary_op)
                    writer.add_summary(summary_str, float(T))
                    last_summary_time = now
                    
            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})
            
            # Choose next action based on e-greedy policy
            a_t = np.zeros([num_actions])
            action_index = 0
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps
            
            
            # Gym excecutes action in game environment on behalf of actor-learner
            s_t1, r_t, terminal, info = env.step(action_index)

            # Accumulate gradients
            readout_j1 = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + FLAGS.gamma * np.max(readout_j1))
    
            a_batch.append(a_t)
            s_batch.append(s_t)
    
            # Update the state and counters
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # Optionally update target network
            if T % FLAGS.target_network_update_frequency == 0:
                session.run(reset_target_network_params)
    
            # Optionally update online network
            if t % FLAGS.network_update_frequency == 0 or terminal:
                if s_batch:
                    loss, _ = session.run([cost, grad_update], feed_dict = {y : y_batch,
                                                          a : a_batch,
                                                          s : s_batch})
                    total_loss += loss
                # Clear gradients
                s_batch = []
                a_batch = []
                y_batch = []
    
            # Save model progress
            if t % FLAGS.checkpoint_interval == 0 and thread_id==(FLAGS.num_concurrent)/2:
                saver.save(session, FLAGS.checkpoint_dir+"/"+FLAGS.experiment+".ckpt", global_step = T)
            
                    
            # Print end of episode stats
            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon, total_loss/float(ep_t)]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ EPSILON PROGRESS", t/float(FLAGS.anneal_epsilon_timesteps), "/ AVERAGE LOSS",  total_loss/float(ep_t)
                break
            

def build_graph(num_actions):
    # Create shared deep q network
    s, q_network = build_network(num_actions=num_actions)
    network_params = q_network.trainable_weights
    q_values = q_network(s)

    # Create shared target network
    st, target_q_network = build_network(num_actions=num_actions)
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)
    
       
    # Op for periodically updating target network with online network weights
    reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
       
    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - action_q_values))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s" : s, 
                 "q_values" : q_values,
                 "st" : st, 
                 "target_q_values" : target_q_values,
                 "reset_target_network_params" : reset_target_network_params,
                 "a" : a,
                 "y" : y,
                 "cost" : cost,
                 "grad_update" : grad_update}

    return graph_ops

# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary(FLAGS.game + ": Episode Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary(FLAGS.game + ": Max Q Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.scalar_summary(FLAGS.game + ": Epsilon", logged_epsilon)
    episode_avg_loss = tf.Variable(0.)
    tf.scalar_summary(FLAGS.game + ": Average Episode Loss", episode_avg_loss)
    logged_T = tf.Variable(0.)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon, episode_avg_loss]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, update_ops, summary_op

def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    # Figure out number of actions from gym env
    env = gym.make(FLAGS.game)
    num_actions = env.action_space.n
    if (FLAGS.game == "Pong-ram-v0" or FLAGS.game == "Breakout-ram-v0"):
        # Gym currently specifies 6 actions for pong
        # and breakout when only 3 are needed. This
        # is a lame workaround.
        num_actions = 3
    return num_actions

def train(session, graph_ops, num_actions, saver):
    global T
    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    # Set up game environments (one per thread)
    envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables for continuing if necessary
    if not FLAGS.cont:
        session.run(tf.initialize_all_variables())
    else:
        f=open("LOG.txt","r")
        lines=f.readlines()
        T=np.int(lines[0])
        print "Continuing from T= ", T
        saver.restore(session, FLAGS.checkpoint_dir + "/FINAL_" + FLAGS.experiment +".ckpt-0")
        f.close()
    
    summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
    writer = tf.train.SummaryWriter(summary_save_path, session.graph)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # Start num_concurrent actor-learner training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, envs[thread_id], session, graph_ops, num_actions, summary_ops, saver)) for thread_id in range(FLAGS.num_concurrent)]
    try:
        for t in actor_learner_threads:
            t.start()

        # Show the agents training and write summary statistics
        #last_summary_time = 0
        #while True:
        #    if FLAGS.show_training:
        #        for env in envs:
        #            env.render()
        #    now = time.time()
        #    if now - last_summary_time > FLAGS.summary_interval:
        #        summary_str = session.run(summary_op)
        #        writer.add_summary(summary_str, float(T))
        #        last_summary_time = now
        signal.pause()
    except (KeyboardInterrupt,SystemExit):
        f=open("LOG.txt","w+")
        f.write('{0}'.format(T))
        f.close()
        saver.save(session, FLAGS.checkpoint_dir+"/FINAL_"+FLAGS.experiment+".ckpt", global_step = 0)
            
def evaluation(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    print "Restored model weights from ", FLAGS.checkpoint_path
    monitor_env = gym.make(FLAGS.game)
    monitor_env.monitor.start(FLAGS.eval_dir+"/"+FLAGS.experiment+"/eval")

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env)
    avg_reward=0
    epsilon=FLAGS.epsilon_test
    num_actions = get_num_actions()
    for i_episode in xrange(FLAGS.num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print ep_reward
        avg_reward+=ep_reward
    avg_reward=avg_reward/FLAGS.num_eval_episodes
    print 'Average Reward RAM-Boltz: ' , avg_reward
    monitor_env.monitor.close()

def main(_):
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    K.set_session(session)
    num_actions = get_num_actions()
    graph_ops = build_graph(num_actions)
    saver = tf.train.Saver(max_to_keep=0)
    session.run(tf.initialize_all_variables())
    #session.run(tf.global_variables_initializer())

    if FLAGS.testing:
        evaluation(session, graph_ops, saver)
    else:
        train(session, graph_ops, num_actions, saver)
            
        

if __name__ == "__main__":
    tf.app.run()
