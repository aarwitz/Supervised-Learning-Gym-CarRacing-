import os
import numpy as np
# import sys

import gym
from pyglet.window import key
import matplotlib.pyplot as plt

def load_imitations(data_folder):
    """
    Given the folder containing the expert imitations, the data gets loaded and
    stored in two lists: observations and actions.

    data_folder: python string, the path to the folder containing the
                observation_%05d.npy and action_%05d.npy files

    return:
    observations: python list of N numpy.ndarrays of size (96, 96, 3)
    actions: python list of N numpy.ndarrays of size 3
    """
    observations = []
    actions = []

    # Get a list of all files in the data folder
    file_list = os.listdir(data_folder)
    file_list.sort()


    for file_name in file_list:
        if file_name.startswith("observation_"):
            # Extract the numerical part from the filename
            observation_fname = file_name
            # print(observation_fname)
            action_fname = file_name.replace("observation","action")
            # print(action_fname)
            # Check if the corresponding action file exists
            if action_fname in file_list:
                # Load and append the observation file
                observation_path = os.path.join(data_folder, observation_fname)
                observation = np.load(observation_path)
                observations.append(observation)

                # Load and append the corresponding action file
                action_path = os.path.join(data_folder, action_fname)
                action = np.load(action_path)
                actions.append(action)
                # if action[1] < 0.5:
                #     print(action[1])
                # if action[1] 
                    # print('ohh')
    # print('observations')
    # print(observations)
    # print('actions')
    # print(actions)
    return observations, actions

def save_imitations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_imitations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    # Ensure the data_folder exists
    print(data_folder)
    if not os.path.exists(data_folder):
        print(data_folder)
        os.makedirs(data_folder)

    # Save actions and observations as .npy files
    for i, (action, observation) in enumerate(zip(actions, observations)):
        action_filename = os.path.join(data_folder, f'action_{i:05d}.npy')
        observation_filename = os.path.join(data_folder, f'observation_{i:05d}.npy')
        np.save(action_filename, action)
        np.save(observation_filename, observation)


class ControlStatus:
    """
    Class to keep track of key presses while recording imitations.
    """

    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def key_press(self, k, mod):
        if k == key.ESCAPE: self.quit = True
        if k == key.SPACE: self.stop = True
        if k == key.TAB: self.save = True
        if k == key.LEFT: self.steer = -1.0
        if k == key.RIGHT: self.steer = +1.0
        if k == key.UP: self.accelerate = +0.5
        if k == key.DOWN: self.brake = +0.8

    def key_release(self, k, mod):
        if k == key.LEFT and self.steer < 0.0: self.steer = 0.0
        if k == key.RIGHT and self.steer > 0.0: self.steer = 0.0
        if k == key.UP: self.accelerate = 0.0
        if k == key.DOWN: self.brake = 0.0


def record_imitations(imitations_folder):
    """
    Function to record own imitations by driving the car in the gym car-racing
    environment.
    imitations_folder:  python string, the path to where the recorded imitations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """
    print('imitations fodleer for recording')
    print(imitations_folder)
    env = gym.make('CarRacing-v0').env
    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()
        env.render()

        # set the functions to be called on key press and key release
        env.viewer.window.on_key_press = status.key_press
        env.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                     status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            total_reward += reward
            env.render()

        if status.save:
            save_imitations(imitations_folder, actions, observations)
            status.save = False

        status.stop = False
        env.close()
        
        
# data_folder = r"/home/aaron/Documents/School/Exercise 1/data/teacher"
# load_imitations(data_folder)
