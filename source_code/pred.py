"""
AUTHOR: Kavya Konisa, Hari Chowdary Madhamanchi
FILENAME: pred.py
SPECIFICATION: In this file, we have tried to predict the carla agent from taking the trained model
FOR: CS 5392 Reinforcement Learning Section 001
"""
import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
#importing the other classes
from constants import *
from car_env import *   #importing carla environment
from tutorial_3 import * 


MODEL_PATH = "C:/Users/balag/OneDrive/Desktop/PP model_log files/models/Xception.model"

if __name__ == '__main__':

    # Memory fraction, used mostly when training multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY)
    tensor_config= tf.ConfigProto(gpu_options=gpu_options)
    tensor_session= tf.Session(config=tensor_config)
    backend.set_session(tensor_session)

    # Load the model
    model_loaded= load_model(MODEL_PATH)
    # Create environment
    carla_env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # It's better to do a first prediction then before we start iterating over episode steps
    model_loaded.predict(np.ones((1, carla_env.HEIGHT, carla_env.WIDTH, 3)))

    # Loop over episodes
    while True:

        print('restart Episodes.')
        # Reset environment and get initial state
        current_state = carla_env.reset()
        carla_env.history_collissions = []
        flag_done = False
        # Loop over steps
        while True:
            # time calculating for appending to fps counter
            step_start = time.time()
            # Showing the current frame
            cv2.imshow(f'Agent', current_state)
            cv2.waitKey(1)
            # Predict an action based on current observation space
            qstate = model_loaded.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qstate)
            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, flag_done, _ = carla_env.step(action)
            #  current state for next loop iteration
            current_state = new_state
            # If flag_done - agent crashed, break an episode
            if flag_done:
                break
            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            fps_counter.append(time.time() - step_start)
            length=len(fps_counter)
            total=sum(fps_counter)
            avg= length/total
            print(f'Agent: {avg:>4.1f} FPS | Action: [{qstate[0]:>5.2f}, {qstate[1]:>5.2f}, {qstate[2]:>5.2f}] {action}')

        # Destroy the carla actor at end of episode
        for actor in carla_env.actor_list:
            actor.destroy()