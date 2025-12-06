import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional, Any, Dict


class CarRacingWrapper(gym.Wrapper):
    """
    Wrapper for CarRacing environment with preprocessing:
    - Grayscale conversion
    - Frame resizing to 84x84
    - Frame stacking (4 frames)
    - Reward shaping
    - Negative reward detection (off-track detection)
    """
    def __init__(
        self, 
        env_name: str = 'CarRacing-v3', 
        frame_stack: int = 4, 
        frame_skip: int = 2, 
        render_mode: Optional[str] = None
    ) -> None:
        """
        Initialize the CarRacing environment wrapper.

        Args:
            env_name: Name of the Gymnasium environment.
            frame_stack: Number of consecutive frames to stack together.
            frame_skip: Number of frames to skip (action repeat).
            render_mode: Rendering mode ('human', 'rgb_array', or None).
        """
        env = gym.make(env_name, render_mode=render_mode)
        super(CarRacingWrapper, self).__init__(env)
        
        self.frame_stack: int = frame_stack
        self.frame_skip: int = frame_skip
        self.frames: deque = deque(maxlen=frame_stack)
        self.reward_threshold: float = -0.1
        self.negative_reward_counter: int = 0
        self.max_negative_steps: int = 50
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for environment reset.

        Returns:
            observation: Stacked frames of shape (frame_stack, 84, 84).
            info: Additional environment information dictionary.
        """
        # Gymnasium returns (obs, info) tuple
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._preprocess(obs)
        
        # Fill frame stack with initial observation
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        
        self.negative_reward_counter = 0
        return self._get_stacked_frames(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action in the environment.

        Args:
            action: Action to execute, numpy array of shape (3,) with [steer, gas, brake].

        Returns:
            observation: Stacked frames of shape (frame_stack, 84, 84).
            reward: Shaped reward for this step.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
            info: Additional environment information dictionary.
        """
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
    
    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale and resize to 84x84.

        Args:
            obs: RGB observation from environment, shape (96, 96, 3).

        Returns:
            Preprocessed grayscale frame, shape (84, 84), normalized to [0, 1].
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized / 255.0
        
        return normalized
    
    def _get_stacked_frames(self) -> np.ndarray:
        """
        Stack frames along channel dimension.

        Returns:
            Stacked frames of shape (frame_stack, 84, 84).
        """
        return np.stack(self.frames, axis=0)
    
    def _shape_reward(self, reward: float) -> float:
        """
        Shape rewards to encourage better behavior.

        Args:
            reward: Raw reward from the environment.

        Returns:
            Shaped reward value.
        """
        if reward < 0:
            # Penalize going off track more heavily
            return reward * 2
        else:
            # Slightly boost positive rewards
            return reward * 1.0
        
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment."""
        return self.env.close()


def make_env(
    env_name: str = 'CarRacing-v3', 
    frame_stack: int = 4, 
    frame_skip: int = 2, 
    render_mode: Optional[str] = None
) -> CarRacingWrapper:
    """
    Create wrapped CarRacing environment.

    Args:
        env_name: Name of the Gymnasium environment.
        frame_stack: Number of consecutive frames to stack together.
        frame_skip: Number of frames to skip (action repeat).
        render_mode: Rendering mode ('human', 'rgb_array', or None).

    Returns:
        Wrapped CarRacing environment.
    """
    return CarRacingWrapper(env_name, frame_stack, frame_skip, render_mode)
