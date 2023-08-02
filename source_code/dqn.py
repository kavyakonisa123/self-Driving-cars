"""
AUTHOR: Gopi Nettam, Abhishek Varma Poosapati
FILENAME: dqn.py
SPECIFICATION: In this file, we have created deep Q network agent that interacts with the environment
FOR: CS 5392 Reinforcement Learning Section 001
"""
import random
import time
import numpy as np

from collections import deque

#importing other classes
from constants import *
from tensorboard_stat import *
#impoting keras
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation,  Flatten
from keras.models import Sequential, Model

import tensorflow as tf



"""
 NAME: DQNAgent
 PURPOSE: The DQNAgent uses a neural network to approximate the Q-values of different actions in a given state, and updates its policy based on feedback in the form of rewards received from the environment.
 INVARIANTS: the DQNAgent class requires memory buffer for experience replay , Q-value update based on the Bellman equation
"""
class DQNAgent:
    """
    NAME: init
    PARAMETERS: self (instance of class)
    PURPOSE: The constructor function for the DQN agent class that initializes the agent's model, target model, replay memory, termination flag, logging variables, TensorBoard, target counter, and TensorFlow graph.
    PRECONDITION: None
    POSTCONDITION: After the function, the DQN agent's instance variables are initialized with default values.
    """
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REP_MSIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_TAKEN}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.initialize_training = False
    """ 
    NAME: calculate_states
    PARAMETERS: minibatch (list): List of transitions containing current states, actions, rewards, and next states.
    PURPOSE: The function calculates the current Q-states and future Q-states from the minibatch.
    PRECONDITION: The minibatch should not be empty
    POSTCONDITION: After the function, it returns current_qstates (numpy array): Predicted Q-values for current states,future_qstates (numpy array): Predicted Q-values for future states.
    """

    def calculate_states(self, minibatch):
        # Convert current states to numpy array and normalize pixel values to [0, 1]
        current_states = np.array([transition[0] for transition in minibatch]) / 255

        # Predict Q-values for current states using the model
        with self.graph.as_default():
            current_qstates = self.model.predict(current_states, PREDICT_BSIZE)

        # Convert next states to numpy array and normalize pixel values to [0, 1]
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255

        # Predict Q-values for future states using the target model
        with self.graph.as_default():
            future_qstates = self.target_model.predict(new_current_states, PREDICT_BSIZE)

        return current_qstates, future_qstates

    """ 
    NAME: create_model_Xception
    PARAMETERS: self, instance of class
    PURPOSE: The function create the Xception base model with random weights , compile the model with MSE loss, Adam optimizer, and accuracy metric
    PRECONDITION: None
    POSTCONDITION: After the function, it returns a compiled Keras model ready for training and prediction.
    """
    def create_model_Xception(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    """ 
    NAME: create_model
    PARAMETERS: self, instance of class
    PURPOSE: The function create the CNN 5 layered base applied activation, averagepooling to each layer, compile the model with MSE loss, Adam optimizer, and accuracy metric
    PRECONDITION: None
    POSTCONDITION: After the function, it returns a compiled Keras model ready for training and prediction.
    Reference:https://github.com/Sentdex/Carla-RL/blob/master/sources/models.py
    """
    def create_model(self):
        input = Input(shape=(IM_HEIGHT, IM_WIDTH,3))
        cnn_1_c1 = Conv2D(64, (7, 7), strides=(3, 3), padding='same')(input)
        cnn_1_a = Activation('relu')(cnn_1_c1)

        cnn_2_c1 = Conv2D(64, (5, 5), strides=(3, 3), padding='same')(cnn_1_a)
        cnn_2_a1 = Activation('relu')(cnn_2_c1)
        cnn_2_c2 = Conv2D(64, (3, 3), strides=(3, 3), padding='same')(cnn_1_a)
        cnn_2_a2 = Activation('relu')(cnn_2_c2)
        cnn_2_ap = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(cnn_1_a)
        cnn_2_c = Concatenate()([cnn_2_a1, cnn_2_a2, cnn_2_ap])

        cnn_3_c1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(cnn_2_c)
        cnn_3_a1 = Activation('relu')(cnn_3_c1)
        cnn_3_c2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(cnn_2_c)
        cnn_3_a2 = Activation('relu')(cnn_3_c2)
        cnn_3_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_2_c)
        cnn_3_c = Concatenate()([cnn_3_a1, cnn_3_a2, cnn_3_ap])

        cnn_4_c1 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(cnn_3_c)
        cnn_4_a1 = Activation('relu')(cnn_4_c1)
        cnn_4_c2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(cnn_3_c)
        cnn_4_a2 = Activation('relu')(cnn_4_c2)
        cnn_4_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_3_c)
        cnn_4_c = Concatenate()([cnn_4_a1, cnn_4_a2, cnn_4_ap])

        cnn_5_c1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(cnn_4_c)
        cnn_5_a1 = Activation('relu')(cnn_5_c1)
        cnn_5_gap = GlobalAveragePooling2D()(cnn_5_a1)
        predictions = Dense(3, activation="linear")(cnn_5_gap)
        model = Model(inputs=input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    """ 
    NAME: create_model_cnn
    PARAMETERS: self, instance of class
    PURPOSE: The function create the CNN  basic model added the convolutions activation etc., compile the model with MSE loss, Adam optimizer, and accuracy metric
    PRECONDITION: None
    POSTCONDITION: After the function, it returns a compiled Keras model ready for training and prediction.
    Reference:https://github.com/Sentdex/Carla-RL/blob/master/sources/models.py
    """
    def create_model_cnn(self):
        base_model = Sequential()

        base_model.add(Conv2D(32, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH,3), padding='same'))
        base_model.add(Activation('relu'))
        base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        base_model.add(Conv2D(64, (3, 3), padding='same'))
        base_model.add(Activation('relu'))
        base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        base_model.add(Conv2D(64, (3, 3), padding='same'))
        base_model.add(Activation('relu'))
        base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        base_model.add(Conv2D(128, (3, 3), padding='same'))
        base_model.add(Activation('relu'))
        base_model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

        base_model.add(Flatten())
        x = base_model.output
        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model  

    """
    NAME: replay_memory_upd
    PARAMETERS: self (instance of class), transition (tuple)
    PURPOSE: The function updates the replay memory with a new transition tuple.
    PRECONDITION: None
    POSTCONDITION: After the function, the new transition tuple is appended to the replay memory.
    """
    def replay_memory_upd(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    """
    NAME: train
    PARAMETERS: self (instance of class)
    PURPOSE: The function performs the training of the DQN agent using the replay memory and updates the target model weights periodically.
    PRECONDITION: The replay memory must have enough transitions to form a minibatch of size MINI_BSIZE.
    POSTCONDITION: After the function, the agent's model and target model weights are updated based on the minibatch of transitions from the replay memory.
    """


    def train(self):
        # Return if replay memory has less samples than the minimum replay memory size required
        if  MIN_REP_MSIZE> len(self.replay_memory):
            return

        minibatch = random.sample(self.replay_memory, MINIB_SIZE)
        current_qstates,future_qstates=self.calculate_states(minibatch)

        X = []  # Initialize input data for training
        y = []  # Initialize target data for training

        for index, (current_state, action, reward, new_state, flag_done) in enumerate(minibatch):
            new_qvalue = self.calculate_newqvalue(future_qstates, index, flag_done, reward)  # Calculate new Q-value using Bellman equation
            current_qs = current_qstates[index]  # Get current Q-values for the state-action pair from the mini-batch
            current_qs[action] = new_qvalue  # Update the Q-value for the selected action with the new Q-value

            X.append(current_state)  # Append current state to input data
            y.append(current_qs)  # Append updated Q-values to target data

        log_this_step = False

        log_this_step = self.log_steps(log_this_step, X, y)  # Log training steps if required

        if self.target_counter > UPDATE:
            self.target_model.set_weights(self.model.get_weights())  # Update the target model weights with the current model weights
            self.target_counter = 0  # Reset the target counter after updating the target model weights


    """
    NAME: calculate_newqvalue
    PARAMETERS: self (instance of class), future_qstates (numpy array), index (int), flag_done (bool), reward (float)
    PURPOSE: The function calculates the new Q-value for a given state-action pair using the Bellman equation.
    PRECONDITION: None
    POSTCONDITION: After the function, the new Q-value is returned based on whether the episode is done or not.
    """
    def calculate_newqvalue(self,future_qstates,index,flag_done,reward):
        # Function to calculate the new Q-value for a given state-action pair

        if not flag_done:
            # If the episode is not done, calculate the maximum Q-value from the future states
            max_future_q = np.max(future_qstates[index])
            # Use the Bellman equation to calculate the new Q-value
            new_qvalue = reward + DISCOUNT * max_future_q
        else:
            # If the episode is done, set the new Q-value to the immediate reward
            new_qvalue = reward

        # Return the calculated new Q-value
        return new_qvalue

    """
    NAME: get_qstate
    PARAMETERS: self (instance of class), state (np.ndarray)
    PURPOSE: Get the Q-values for a given state from the DQN agent's model.
    PRECONDITION: The state should be a valid numpy array.
    POSTCONDITION: Returns the Q-values for the given state.
    """
    def get_qstate(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    """
    NAME: fit_data
    PARAMETERS: self (instance of class)
    PURPOSE: Perform a single fitting step of the DQN agent's model with random data.
    PRECONDITION: None
    POSTCONDITION: The model's weights may be updated based on the fitting step.
    """
    def fit_data(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

    """
    NAME: train_in_loop
    PARAMETERS: self (instance of class)
    PURPOSE: Train the DQN agent's model in a loop until terminated.
    PRECONDITION: None
    POSTCONDITION: The model's weights may be updated during training.
    """
    def train_in_loop(self):
        self.fit_data()        

        self.initialize_training = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


    """
    NAME: log_steps
    PARAMETERS: self (instance of class), log_this_step (bool), X (list of np.ndarray), y (list of np.ndarray)
    RETURNS: bool
    PURPOSE: Log training steps to Tensorboard and update target model weights if required.
    PRECONDITION: X and y should contain valid numpy arrays representing input and target data for training.
    POSTCONDITION: The training steps may be logged to Tensorboard and target model weights may be updated.
    """
    def log_steps(self,log_this_step,X,y):
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINB_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_counter += 1

        return log_this_step