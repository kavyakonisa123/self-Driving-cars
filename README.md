# Autonomous Driving with Deep Reinforcement Learning

This document provides instructions for setting up and running CARLA (Car Learning to Act) on Windows. Before proceeding, ensure you have Python version 3.7 or later, Carla software version 0.9.5, and Unreal Engine installed on your system.

## About
The goal is to develop a self-driving agent that can steer a vehicle in the CARLA simulator. The agent is trained using Q-Learning and Deep Q-Networks (DQN) to optimize its driving policy based on rewards from the environment.

CARLA provides a realistic 3D urban driving simulator for testing and training autonomous driving systems. It allows configuring various aspects like weather, traffic, and road conditions.

## Step 1: Install Dependencies

1. Install Unreal Engine: Download and install Unreal Engine version 4.24, 4.25, or 4.26 from the official Unreal Engine website [4].

2. Install Python: Download and install Python 3.7 or later (64-bit version) from the official Python website.

3. Install CUDA Toolkit and cuDNN (optional): If you plan to use GPU acceleration for training, you will need to install CUDA Toolkit and cuDNN as per NVIDIA's official documentation.

## Step 2: Set up Unreal Engine

1. Open Unreal Engine and create a new project.

2. Close the Unreal Engine editor after the project is created.

## Step 3: Clone CARLA Repository

1. Open Git Bash or any other Git client on your Windows machine.

2. Clone the CARLA repository from GitHub using the following command:
   ```
   git clone https://github.com/carla-simulator/carla.git
   ```

3. Change directory to the CARLA repository:
   ```
   cd carla
   ```

Alternatively, you can download the CARLA software from the official website [1] or use the Carla package uploaded to our drive [3]. If you use the Carla package from the drive, extract the files and find the source code in the `pythonAPI/examples/` directory.

## Installation and Execution

1. Upgrade pip:
   ```
   python -m pip install --upgrade pip
   ```

2. Install the necessary libraries from the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

3. Double-click the CarlaUE4 application in the Carla folder to launch the Carla environment on Windows. For Linux, use the terminal and execute the simulator with the command:
   ```
   ./CarlaUE4.sh
   ```

4. To execute the developed code, navigate to the `PythonAPI/examples` directory and run the desired file using Python. For example:
   ```
   python tutorial_3.py
   ```

## Models

We have worked on three models: a 5-layered CNN, Xception, and a simple CNN model. To train a specific model, update the function name accordingly in `dqn.py`. For example, if you want to train on Xception, rename the `create_model` function to `train_Xception`.

## File Overview

- All constants are declared in `constants.py`.
- The Carla environment agent is implemented in `car_env.py`.
- The DQN agent is implemented in `dqn.py`.
- These components are integrated in `tutorial_3.py`.

## References

- [1] CARLA Documentation: https://carla.readthedocs.io/en/latest/start_quickstart/
- [2] Tensorboard: https://github.com/Noam2710/deep-q-learning/blob/master/ModifiedTensorBoard.py
- [3] Carla Package: https://drive.google.com/drive/folders/14hBbkOpA8PbkLc7JKArTQr8L_zz2c9-O?usp=share_link
- [4] Unreal Engine: https://www.unrealengine.com/en-US/ue-on-github
- [5] What is pip: https://realpython.com/what-is-pip/

