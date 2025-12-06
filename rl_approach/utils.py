import os
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any


class Logger:
    """
    Simple logger for training metrics.
    """
    def __init__(self, log_dir: str = 'logs') -> None:
        """
        Initialize the logger.

        Args:
            log_dir: Directory to save log files.
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        with open(self.log_file, 'w') as f:
            f.write('Episode,Steps,Reward,AvgReward,PolicyLoss,ValueLoss,EntropyLoss\n')
    
    def log(self, episode: int, steps: int, reward: float, avg_reward: float, 
            losses: Optional[Dict[str, float]]) -> None:
        """
        Log training metrics.

        Args:
            episode: Episode number.
            steps: Number of steps taken in the episode.
            reward: Total reward for the episode.
            avg_reward: Moving average reward.
            losses: Dictionary of loss values (policy_loss, value_loss, entropy_loss) or None.
        """
        log_str = f"Episode {episode:4d} | Steps: {steps:5d} | Reward: {reward:8.2f} | Avg Reward: {avg_reward:8.2f}"
        
        if losses:
            log_str += f" | Policy Loss: {losses['policy_loss']:.4f} | Value Loss: {losses['value_loss']:.4f} | Entropy: {losses['entropy_loss']:.4f}"
        
        print(log_str)
        
        with open(self.log_file, 'a') as f:
            if losses:
                f.write(f"{episode},{steps},{reward:.2f},{avg_reward:.2f},{losses['policy_loss']:.4f},{losses['value_loss']:.4f},{losses['entropy_loss']:.4f}\n")
            else:
                f.write(f"{episode},{steps},{reward:.2f},{avg_reward:.2f},,,\n")


class RewardTracker:
    """
    Track and compute moving average of rewards.
    """
    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize the reward tracker.

        Args:
            window_size: Number of recent episodes to use for computing moving average.
        """
        self.rewards: List[float] = []
        self.window_size = window_size
    
    def add(self, reward: float) -> None:
        """
        Add a reward to the tracker.

        Args:
            reward: Reward value to add.
        """
        self.rewards.append(reward)
    
    def get_average(self) -> float:
        """
        Get the moving average reward.

        Returns:
            Moving average of the last window_size rewards.
        """
        if len(self.rewards) == 0:
            return 0.0
        return np.mean(self.rewards[-self.window_size:])
    
    def get_all(self) -> List[float]:
        """
        Get all stored rewards.

        Returns:
            List of all reward values.
        """
        return self.rewards


def save_checkpoint(agent: Any, episode: int, reward: float, filepath: str) -> None:
    """
    Save training checkpoint.

    Args:
        agent: PPO agent to save.
        episode: Current episode number.
        reward: Current reward (for logging).
        filepath: Path to save the checkpoint file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    agent.save(filepath)
    print(f"Checkpoint saved: Episode {episode}, Reward {reward:.2f}")


def load_checkpoint(agent: Any, filepath: str) -> bool:
    """
    Load training checkpoint.

    Args:
        agent: PPO agent to load checkpoint into.
        filepath: Path to the checkpoint file.

    Returns:
        True if checkpoint was loaded successfully, False otherwise.
    """
    if os.path.exists(filepath):
        agent.load(filepath)
        return True
    else:
        print(f"Checkpoint not found: {filepath}")
        return False
