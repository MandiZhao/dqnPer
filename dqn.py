import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
from atari_wrappers import _process_frame84

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    total_timestep=2e8, ##see LinearSchedule param above 
    explo_frac=0.1, ##see LinearSchedule param above 
    stopping_criterion=None,
    replay_buffer_size=500000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=1,  ##replay freq has to be 1 b.c. need update priority for each replay?
    frame_history_len=4,
    target_update_freq=5000, ##baseline only uses 500
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False,
    alpha=0.6,
    beta0=0.4,
    beta_iters=2e8,
    replay_eps=1e-6
    ):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    ***added for prioritized replay**
    alpha: float
    beta0: float
        initial beta value for buffer
    beta_iters: int
        number of iters over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps <<<<<<????
    replay_eps: float
        epsilon to add to the TD errors when updating priorities

    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file

    ###############
    # BUILD MODEL #
    ###############

    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = self.env.observation_space.shape
    else:
        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    self.num_actions = self.env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    self.obs_t_ph              = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    self.obs_tp1_ph            = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])

    ## Additionally for PER
    self.importance_weights_ph = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    if lander:
      obs_t_float = self.obs_t_ph
      obs_tp1_float = self.obs_tp1_ph
    else:
      obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

    # Compute the Bellman error. 
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network
    ######

    # YOUR CODE HERE
    q = q_func(obs_t_float, self.num_actions, scope='q_func', reuse=False)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q = q_func(obs_tp1_float, self.num_actions, scope='target_q_func', reuse=False)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')
    self.predicted_best_act = tf.argmax(q, axis=1)
    if not double_q:
        q_t = tf.reduce_max(target_q, axis=1)
    else:
        print("not implemented")

    y = self.rew_t_ph + (1.0 - self.done_mask_ph) * gamma * q_t
    y_pred = tf.reduce_sum(q * tf.one_hot(self.act_t_ph, self.num_actions), axis=1)
    self.total_error = 0.5 * tf.reduce_mean(huber_loss(y_pred - tf.stop_gradient(y)))
    self.td_error = y_pred - tf.stop_gradient(y)
    self.weighted_error = tf.reduce_mean(self.importance_weights_ph * huber_loss(y_pred - tf.stop_gradient(y)))


    ######

    # construct optimization op (with gradient clipping)
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    self.train_fn = minimize_and_clip(optimizer, self.weighted_error, 
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    self.update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer 
    ## ** Now needs extra ALPHA param
    self.replay_eps = replay_eps
    self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, frame_history_len, alpha, lander)
    self.replay_buffer_idx = None
    if beta_iters is None:
        beta_iters = total_timestep
    self.beta_schedule = LinearSchedule(beta_iters, final_p=1.0, initial_p=beta0)

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.last_obs = _process_frame84(self.last_obs)
    self.log_every_n_steps = 100000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # 
    # Specifically, self.last_obs must point to the new latest observation.
    # Call env.reset() to get a new observation if done is true
    # Include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####

    # YOUR CODE HERE
    idx = self.replay_buffer.store_frame(self.last_obs)
    obs = self.replay_buffer.encode_recent_observation() ##now it's the most recent `frame_history_len` frames.

    if self.model_initialized and random.random() > self.exploration.value(self.t):
        act_val = self.session.run(self.predicted_best_act, feed_dict={self.obs_t_ph:[obs]})
        act = act_val[0]
    else:
        act = self.env.action_space.sample()

    obs, reward, done, info = self.env.step(act)
    self.last_obs = _process_frame84(obs)  ##self.env.step is not producing proper shape either
    self.replay_buffer.store_effect(idx, act, reward, done)
    if done:
        self.last_obs = self.env.reset()
        self.last_obs = _process_frame84(self.last_obs)

  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # Needs replay buffer contain enough samples, until then, the model will not be
    # initialized and random actions should be taken

    ## learning_freq = freq to sample batch from buffer = "K" in PER paper
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
        # Training consists of four steps:
        # 3.a: use the replay buffer to sample a batch of transitions 
        # 3.b: initialize the model if it has not been initialized yet; 
        # Remember that you have to update the target network too (see 3.d)!
        # 3.c: train the model. 
        # Use self.train_fn and
        # self.total_error ops that were created earlier: self.total_error is what you
        # created to compute the total Bellman error in a batch, and 
        # self.train_fn will actually perform a gradient step and update the network parameters
        # to reduce total_error. 
        # 3.d: periodically update the target network by calling
        #####

        # YOUR CODE HERE
        # 3.a sample a batch AND update prorities 
        sample_ret = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(self.t))
        ##now needs beta param
        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask, weights, batch_idxes = sample_ret


        # 3.b: initialize the model
        if not self.model_initialized:
            initialize_interdependent_variables(self.session, tf.global_variables(), {
            self.obs_t_ph: obs_t_batch, self.obs_tp1_ph: obs_tp1_batch})
            _ = self.session.run(self.update_target_fn)
            self.model_initialized = True

        # 3.c: train the model
        feed_dict = {
            self.obs_t_ph: obs_t_batch,
            self.act_t_ph: act_batch,
            self.rew_t_ph: rew_batch,
            self.obs_tp1_ph: obs_tp1_batch,
            self.done_mask_ph: done_mask,
            self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t),
            self.importance_weights_ph: weights
        }
        _ = self.session.run([self.train_fn, self.td_error, self.weighted_error], feed_dict=feed_dict)
        
        ### *** ERROR AREA

        v = self.session.run(self.td_error, feed_dict=feed_dict)
        
        new_priorities = np.abs(v) + self.replay_eps
        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        # 3.d periodically update the target network
        if self.num_param_updates % self.target_update_freq == 0:
            _ = self.session.run(self.update_target_fn)
        self.num_param_updates += 1

    self.t += 1

  def log_progress(self):
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))

      self.start_time = time.time()

      sys.stdout.flush()

      with open(self.rew_file, 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()
