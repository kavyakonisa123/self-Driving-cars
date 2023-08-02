"""
AUTHOR: Kavya Konisa, Hari Chowdary Madhamanchi
FILENAME: car_env.py
SPECIFICATION: In this file, we have used the carla simulator to create carla Environment agent with sensors .
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
from dqn import *  #importing dnq agent


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
 NAME: CarEnv
 PURPOSE: The CarEnv encapsulates the environment logic, provide methods for interacting with the environment, and allow the RL agent to observe, take actions, and receive feedback in the form of rewards.
 INVARIANTS: the CarEnv class should define a step() and reset() functions.Define a reward function, state space actionspace etc.
"""
class CarEnv:
     # Constants for camera display and steering amount
    WIDTH = IM_WIDTH       # Width of the image
    HEIGHT = IM_HEIGHT     # Height of the image
    SHOW_CAM = SHOW_CAMERA  # Flag to display camera
    STEER_AMT = 1.0        # Steering amount for control
    # Initialize front camera variable
    front_camera = None    # Variable to hold the front camera image

    """ 
    NAME: __init__
    PARAMETERS: self,  refers to the instance of an object being created or operated upon within a class. 
    PURPOSE: The function is used to connect to the carla simulator, configure and add blueprints
    PRECONDITION: None
    POSTCONDITION: After the function, it creates a instance to the class Env
    """
    def __init__(self):#To connect to a simulator we need to create a "Client" object, we need to provide  IP address and port of a running instance of the simulator
        self.client = carla.Client("localhost", 2000)
        #  time-out sets a time limit to all networking operations.
        self.client.set_timeout(2000.0)
         # once the client configured, we can retrive the world directly.
        self.world = self.client.get_world()
        # blueprint has the necessory information to create an actor, if we define blueprint as a car, we can change color of the car.
        self.blueprint_library = self.world.get_blueprint_library()
        # default blueprint of BMW vehicle.    
        self.bmw = self.blueprint_library.filter("bmw")[0]
    """ 
    NAME: set_old_reward
    PARAMETERS: len_colli, length of the collision history ; velocity , velocity to convert into kmh
    PURPOSE: The function  Function to apply the control action to the vehicle based on the given action
    PRECONDITION: The len_colli, velocity should not be empty
    POSTCONDITION: After the function, it returns flag_done, reward considered for each case
    """
    def set_old_reward(self,len_colli,velocity):
        #calculate the velocity magnitude
        velocity_mag_square = velocity.x**2 + velocity.y**2 + velocity.z**2
        #Convert into velocity to kmh 
        kmh = int(3.6 * math.sqrt(velocity_mag_square))
        # if there are collissions we consider the leat reward of -200 and makes the episode as done in that epoch
        if len_colli != 0:
            flag_done = True
            reward = -200
        # if the speed is less than we consider the reward -1
        elif kmh < 50:
            flag_done = False
            reward = -1
        else:
            flag_done = False
            reward = 1
        return flag_done, reward

    """ 
    NAME: set_reward
    PARAMETERS: len_colli, length of the collision history ; velocity , velocity to convert into kmh
    PURPOSE: The function  Function to apply the control action to the vehicle based on the given action
    PRECONDITION: The len_colli, velocity should not be empty
    POSTCONDITION: After the function, it returns flag_done, reward considered for each case
    ---Have used the research paper to change the reward system criteria
    """
    def set_reward(self,len_colli,velocity):
        #calculate the velocity magnitude
        velocity_mag_square = velocity.x**2 + velocity.y**2 + velocity.z**2
        #Convert into velocity to kmh 
        kmh = int(3.6 * math.sqrt(velocity_mag_square))
        # if there are collissions we consider the leat reward of -200 and makes the episode as done in that epoch
        if len_colli != 0:
            flag_done = True
            reward = -200
        # if the speed is less than we consider the reward -1
        elif kmh < 50:
            flag_done = False
            reward = -1
        else:
            flag_done = False
            reward = 200
        return flag_done, reward

    """ 
    NAME: apply_control_action
    PARAMETERS: action, the action or choice made by an agent at a particular state in an environment.
    PURPOSE: The function  Function to apply the control action to the vehicle based on the given action
    PRECONDITION: The action should not be empty
    POSTCONDITION: After the function, the image doe.s not return anything but applies to the vehicle
    """
    def apply_control_action(self, action):
        # Function to apply the control action to the vehicle based on the given action
        if action == 0:
            # If action is 0, apply control for turning left by setting negative steer amount and full throttle
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            # If action is 1, apply control for going straight by setting zero steer amount and full throttle
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            # If action is 2, apply control for turning right by setting positive steer amount and full throttle
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

    """ 
    NAME: reset
    PARAMETERS: self, instance of class
    PURPOSE: The function function resets the environment to an initial state or a new episode based on the done flag
    PRECONDITION: None
    POSTCONDITION: After the function, the vehicle is resetted with a camera sensor, collission sensor and spawned at a particular point
    """
    def reset(self):
        self.history_collisions = []
        self.actor_list = []

        
         #spawn actor nothing but the vehicle as the simulator has hundrads of spwan points we picked one .
        self.spawn_actor_point = random.choice(self.world.get_map().get_spawn_points())
        # spawn the vehicle.
        self.vehicle = self.world.spawn_actor(self.bmw, self.spawn_actor_point)
        # add the vehicle to the list of actors, as we need to track it and also clean it.
        self.actor_list.append(self.vehicle)
    
        # sensors are actors that provides a continuous stream of data.
        # add blueprint for the sensors and set attributes.
        self.camera_default_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_default_bp.set_attribute("image_size_x", f"{self.WIDTH}")
        self.camera_default_bp.set_attribute("image_size_y", f"{self.HEIGHT}")
        self.camera_default_bp.set_attribute("fov", f"110") #feild of view
        # adusting sensors to vehicel.
        relative_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.camera_default_bp, relative_transform, attach_to=self.vehicle)
         # add sensor to list of actors.
        self.actor_list.append(self.sensor)
        # so we are taking the data from the sensors and we need to get an image of it, so we are using listen function.
        # lamda function used to evaluate the data. 
        self.sensor.listen(lambda data: self.image_processor(data))
        
        # vehicles are one of the type of actor that has extra methods such as  break, throttle, steer values, etc.
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(50)

        # add blueprint for the collision sensors and attach to vehicle
        collisson_sensor = self.blueprint_library.find("sensor.other.collision")
        self.coll_sensor = self.world.spawn_actor(collisson_sensor, relative_transform, attach_to=self.vehicle)
        # add sensor to list of actors.
        self.actor_list.append(self.coll_sensor)
        # lamda function used to add the history of events or collissions
        self.coll_sensor.listen(lambda event: self.collision_data(event))


        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    """ 
    NAME: collision_data
    PARAMETERS: event, data captured by a collision sensor
    PURPOSE: The function appends the sent data to the history_collisions variable
    PRECONDITION: The event should be none/null
    POSTCONDITION: After the function, the returns the history_collisions with new data.
    """
    def collision_data(self, event):
        self.history_collisions.append(event)

    """ 
    NAME: image_processor
    PARAMETERS: image, image captured by a sensor
    PURPOSE: The function first converts into array format and later preprocesses the image based on the given height and width of the image.It returns the preprocessed image
    PRECONDITION: The image should be none/null
    POSTCONDITION: After the function, the image it returns should be preprocessed as per the image.
    """
    def image_processor(self,image):
        # Convert raw image data to a numpy array and reshape it to the image dimensions
        reshape = np.array(image.raw_data).reshape((self.HEIGHT, self.WIDTH, 4))
        # Extract the RGB channels and discard the alpha channel
        remove_alpha = reshape[:, :, :3]

        if self.SHOW_CAM:
            cv2.imshow("", remove_alpha)
            cv2.waitKey(1)
        self.front_camera = remove_alpha
    """ 
    NAME: step
    PARAMETERS: action, the action or choice made by an agent at a particular state in an environment.
    PURPOSE: The function applies action and then sets reward based on velocity.Moreover decides the whether the episode is done or not
    PRECONDITION: The action should not be empty
    POSTCONDITION: After the function, the image it returns the view of front_camera, reward, flag_done
    """
    def step(self, action):
    
        # Function to take a step in the RL environment with the given action

        self.apply_control_action(action)  # Apply the control action to the vehicle
        velocity = self.vehicle.get_velocity()  # Get the current velocity of the vehicle
        flag_done, reward = self.set_reward(self.history_collisions, velocity)  # Set the reward and done flag based on the history of collisions and current velocity
        time_considered = self.episode_start +SEC_EPI  # Calculate the time considered for the episode

        if time_considered < time.time():
            # If the time considered for the episode has elapsed, set the done flag to True
            flag_done = True

        return self.front_camera, reward, flag_done, None  # Return the current front camera image, reward, done flag, and None for info



