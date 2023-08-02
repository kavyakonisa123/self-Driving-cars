"""
AUTHOR: Kavya Konisa, Hari Chowdary Madhamanchi , Gopi Nettam, Abhishek Varma Poosapati,Dheeraj Chowdary Yalamanchili
FILENAME: constants.py
SPECIFICATION: In this file, we have used assigned all the global varioables
FOR: CS 5392 Reinforcement Learning Section 001
"""

SHOW_CAMERA = False  # Flag to indicate whether to show camera feed or not
IM_WIDTH = 640  # Width of the camera feed images
IM_HEIGHT = 480  # Height of the camera feed images


REP_MSIZE = 5_000  # Size of the replay memory buffer
MIN_REP_MSIZE = 1_000  # Minimum size of the replay memory buffer

PREDB_SIZE = 1 # Batch size for predicting actions from the DQN model
MINIB_SIZE = 16# Mini-batch size for training the DQN model
TRAINB_SIZE = MINIB_SIZE // 4 # Batch size for training the DQN model

UPDATE = 5  # Number of episodes before updating the target network
MODEL_TAKEN = "Xception"  # Name of the DQN model

MEMORY = 0.4  # Fraction of GPU memory used for training multiple agents
MIN_REWARD = -200  # Minimum reward value for clipping during training

EPISODES = 100  # Number of episodes for training
DISCOUNT = 0.99  # Discount factor for calculating future rewards
SEC_EPI = 10  # Maximum duration of each episode in seconds

epsilon = 1  # Initial epsilon value for epsilon-greedy exploration
MIN_EPSILON = 0.001  # Minimum epsilon value for exploration
EPSILON_DECAY = 0.95  # Epsilon decay rate for exploration

STATS=10  # Number of episodes to aggregate statistics for logging
