import os
import numpy as np
import torch
from datetime import datetime


class Logger:
    """
    Simple logger for training metrics.
    """
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        with open(self.log_file, 'w') as f:
            f.write('Episode,Steps,Reward,AvgReward,PolicyLoss,ValueLoss,EntropyLoss\n')
    
    def log(self, episode, steps, reward, avg_reward, losses):
        """
        Log training metrics.
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
    def __init__(self, window_size=100):
        self.rewards = []
        self.window_size = window_size
    
    def add(self, reward):
        self.rewards.append(reward)
    
    def get_average(self):
        if len(self.rewards) == 0:
            return 0.0
        return np.mean(self.rewards[-self.window_size:])
    
    def get_all(self):
        return self.rewards


def save_checkpoint(agent, episode, reward, filepath):
    """
    Save training checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    agent.save(filepath)
    print(f"Checkpoint saved: Episode {episode}, Reward {reward:.2f}")


def load_checkpoint(agent, filepath):
    """
    Load training checkpoint.
    """
    if os.path.exists(filepath):
        agent.load(filepath)
        return True
    else:
        print(f"Checkpoint not found: {filepath}")
        return False
