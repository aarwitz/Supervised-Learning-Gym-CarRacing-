import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.
    """
    def __init__(
        self, 
        network,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        mini_batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.network = network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        
        self.memory = RolloutBuffer()
        
    def select_action(self, state, deterministic=False):
        """
        Select action given state and also return log probability and value.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, value = self.network.act(state_tensor, deterministic)
            # Evaluate log_prob for the sampled action
            log_prob, _, _ = self.network.evaluate_actions(state_tensor, action)
        
        return action, value, log_prob
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """
        Store transition in memory.
        """
        self.memory.add(state, action, reward, value, log_prob, done)
    
    def update(self):
        """
        Update policy using PPO.
        """
        # Get data from memory
        states, actions, rewards, values, log_probs, dones = self.memory.get()
        
        if len(states) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device).unsqueeze(-1)
        
        # Compute bootstrap last value for unfinished rollout
        # If the last step was not terminal, use network value as next_value for GAE
        try:
            last_state = self.memory.states[-1]
            last_state_tensor = torch.FloatTensor(np.array(last_state)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.network.get_value(last_state_tensor).cpu().numpy()[0][0]
        except Exception:
            last_value = 0.0
        
        # Compute returns and advantages (pass bootstrap last_value)
        returns, advantages = self._compute_gae(rewards, values, dones, last_value)
        # Convert to tensors and ensure shapes: returns should be (N,1) to match critic output
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(-1)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        update_count = 0
        
        for _ in range(self.ppo_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                if len(batch_indices) < 2:
                    continue
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                log_probs, state_values, entropy = self.network.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Calculate ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss: use SmoothL1Loss (Huber) for stability
                value_loss = nn.SmoothL1Loss()(state_values, batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.value_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                update_count += 1
        
        # Clear memory
        self.memory.clear()
        
        # Return average losses
        if update_count > 0:
            return {
                'policy_loss': total_policy_loss / update_count,
                'value_loss': total_value_loss / update_count,
                'entropy_loss': total_entropy_loss / update_count
            }
        else:
            return {
                'policy_loss': 0,
                'value_loss': 0,
                'entropy_loss': 0
            }
    
    def _compute_gae(self, rewards, values, dones, last_value=0.0):
        """
        Compute Generalized Advantage Estimation (GAE) with optional bootstrap.
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = last_value
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[step + 1]
            
            next_non_terminal = 1.0 - float(dones[step])
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return returns, advantages
    
    def save(self, filepath):
        """
        Save model.
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones
        )
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)
