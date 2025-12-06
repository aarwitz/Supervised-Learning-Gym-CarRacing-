import gymnasium as gym
import numpy as np
import cv2
from collections import deque


class CarRacingWrapper(gym.Wrapper):
    """
    Wrapper for CarRacing environment with preprocessing:
    - Grayscale conversion
    - Frame resizing to 84x84
    - Frame stacking (4 frames)
    - Reward shaping
    - Negative reward detection (off-track detection)
    """
    def __init__(self, env_name='CarRacing-v3', frame_stack=4, frame_skip=2, render_mode=None):
        env = gym.make(env_name, render_mode=render_mode)
        super(CarRacingWrapper, self).__init__(env)
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frame_stack)
        self.reward_threshold = -0.1
        self.negative_reward_counter = 0
        self.max_negative_steps = 50
        
    def reset(self, seed=None, options=None):
        # Gymnasium returns (obs, info) tuple
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._preprocess(obs)
        
        # Fill frame stack with initial observation
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        
        self.negative_reward_counter = 0
        return self._get_stacked_frames(), info
    
    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False
        
        # Frame skipping for faster training
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Preprocess observation
        obs = self._preprocess(obs)
        self.frames.append(obs)
        
        # Reward shaping
        shaped_reward = self._shape_reward(total_reward)
        
        # Detect if car is stuck/off-track
        if total_reward < self.reward_threshold:
            self.negative_reward_counter += 1
        else:
            self.negative_reward_counter = 0
        
        if self.negative_reward_counter >= self.max_negative_steps:
            terminated = True
            shaped_reward -= 10  # Penalty for going off track
        
        return self._get_stacked_frames(), shaped_reward, terminated, truncated, info
    
    def _preprocess(self, obs):
        """
        Convert to grayscale and resize to 84x84.
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized / 255.0
        
        return normalized
    
    def _get_stacked_frames(self):
        """
        Stack frames along channel dimension.
        """
        return np.stack(self.frames, axis=0)
    
    def _shape_reward(self, reward):
        """
        Shape rewards to encourage better behavior.
        - Penalize negative rewards more heavily
        - Encourage staying on track
        """
        if reward < 0:
            # Penalize going off track more heavily
            return reward * 2
        else:
            # Slightly boost positive rewards
            return reward * 1.0
        
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()


def make_env(env_name='CarRacing-v3', frame_stack=4, frame_skip=2, render_mode=None):
    """
    Create wrapped environment.
    """
    return CarRacingWrapper(env_name, frame_stack, frame_skip, render_mode)
