"""
AUTHOR: Kavya Konisa, Hari Chowdary Madhamanchi ,Dheeraj Chowdary Yalamanchili
FILENAME: tutorial_2.py
SPECIFICATION: In this file, we have used the threading concept thread for running the training and inference processes concurrently in separate threads within the same Python process
FOR: CS 5392 Reinforcement Learning Section 001
"""
# importing required libraries.
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

#importing the other classes
from constants import *
from tensorboard_stat import * # tensorboard
from dqn import *  #importing dqn agent
from car_env import *   #importing carla environment


import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

"""
NAME: create_models_folder
PARAMETERS: None
PURPOSE: Create a directory 'models' if it does not exist.
PRECONDITION: None
POSTCONDITION: A directory 'models' may be created if it does not exist.
"""
def create_models_folder():
    if not os.path.isdir('models'):
        os.makedirs('models')

"""
NAME: calculate_action
PARAMETERS: None
RETURNS: action (int) - calculated action
PURPOSE: Calculate the action to be taken based on the epsilon-greedy exploration strategy.
PRECONDITION: epsilon, current_state, dqn_agent, FPS are defined.
POSTCONDITION: The action to be taken is returned.
"""
def calculate_action():
    # Generate a random number
    random_num = np.random.random()
    # Checking if the random number is greater than epsilon
    if epsilon < random_num:      
        # Get action from Q table
        action = np.argmax(dqn_agent.get_qstate(current_state))
    else:
        # Get random action
        action = np.random.randint(0, 3)
        # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
        time.sleep(1/FPS)
    return action

"""
NAME: calculate_reward
PARAMETERS: ep_rewards (list) - list of episode rewards
RETURNS: average_reward (float) - average reward over the last AGG_ALL_STATS episodes
            min_reward (float) - minimum reward over the last AGG_ALL_STATS episodes
            max_reward (float) - maximum reward over the last AGG_ALL_STATS episodes
PURPOSE: Calculate the average, minimum, and maximum rewards over the last AGG_ALL_STATS episodes.
PRECONDITION: ep_rewards, AGG_ALL_STATS are defined.
POSTCONDITION: The average, minimum, and maximum rewards are returned.
"""
def calculate_reward(ep_rewards):
    # Using list slicing to extract the last AGG_ALL_STATS elements from ep_rewards
    recent_ep_rewards = ep_rewards[-STATS:]

    # Calculating the sum, length, minimum, and maximum values of recent_ep_rewards
    total_reward = sum(recent_ep_rewards)
    num_rewards = len(recent_ep_rewards)
    min_reward = min(recent_ep_rewards)
    max_reward = max(recent_ep_rewards)

    # Calculating the average reward by dividing the total reward by the number of rewards
    average_reward = total_reward / num_rewards
    
    return average_reward,min_reward,max_reward


"""
NAME: repeat_conditions
PARAMETERS: None
PURPOSE: Set random seeds for reproducibility in random number generation.
PRECONDITION: random, np, and tf modules are imported.
POSTCONDITION: random seeds are set for random, np, and tf modules.
"""
def repeat_conditions():
    random_seed= 1
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

"""
NAME: actor_destroy
PARAMETERS: None
PURPOSE: Destroy actors in the carla_env.actor_list and remove them from the simulator.
PRECONDITION: carla_env and actor_list are defined.
POSTCONDITION: Actors in actor_list are destroyed and removed from the simulator.
"""
def actor_destroy():
    for actor in carla_env.actor_list:
        actor.destroy()  # remove the actor from the simulator.

if __name__ == '__main__':
    FPS = 60
    ep_rewards = [-200]

    repeat_conditions()

    # Memory fraction, used mostly when training multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY)
    tensor_config= tf.ConfigProto(gpu_options=gpu_options)
    tensor_session= tf.Session(config=tensor_config)
    backend.set_session(tensor_session)
    create_models_folder()

    # Create agent and environment
    dqn_agent = DQNAgent()
    carla_env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=dqn_agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not dqn_agent.initialize_training:
        time.sleep(10)

    # It's better to do a first prediction then before we start iterating over episode steps
    dqn_agent.get_qstate(np.ones((carla_env.HEIGHT, carla_env.WIDTH, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            carla_env.history_collisions = []
            # Update tensorboard step every episode
            dqn_agent.tensorboard.step = episode
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1
            # Reset environment and get initial state
            current_state = carla_env.reset()
            # Reset flag and start iterating until episode ends
            flag_done = False
            episode_start = time.time()
            # Play for given number of seconds only
            while True:
                action = calculate_action()
                new_state, reward, flag_done, _ = carla_env.step(action)
                # Transform new continous state to new discrete state and count reward
                episode_reward += reward
                # Every step we update replay memory
                dqn_agent.replay_memory_upd((current_state, action, reward, new_state, flag_done))
                current_state = new_state
                step += 1
                if flag_done:
                    break

            # End of episode - destroy agents
            actor_destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % STATS or episode == 1:
                average_reward,min_reward,max_reward = calculate_reward(ep_rewards)
                dqn_agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    dqn_agent.model.save(f'models/{MODEL_TAKEN}.model')


            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    dqn_agent.terminate = True
    trainer_thread.join()
    dqn_agent.model.save(f'models/{MODEL_TAKEN}.model')



    