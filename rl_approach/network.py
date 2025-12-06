import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    - Actor outputs the mean and log-standard-deviation parameters of a Gaussian 
    distribution over continuous actions (steer, gas, brake).
    - Critic outputs state value, V(s).
    """
    def __init__(self, num_inputs=4, num_actions=3):
        super(ActorCritic, self).__init__()
        
        # Shared CNN feature extractor
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Shared fully connected layer
        self.fc_shared = nn.Linear(linear_input_size, 512)
        
        # Actor head (policy) - outputs mean and log_std for continuous actions
        self.fc_actor = nn.Linear(512, 256)
        self.action_mean = nn.Linear(256, num_actions)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_actions))
        
        # Critic head (value function)
        self.fc_critic = nn.Linear(512, 256)
        self.value = nn.Linear(256, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize action mean with smaller values
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, 0)
    
    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        # Actor (policy)
        actor_x = F.relu(self.fc_actor(x))
        action_mean = self.action_mean(actor_x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic (value)
        critic_x = F.relu(self.fc_critic(x))
        value = self.value(critic_x)
        
        return action_mean, action_std, value
    
    def act(self, state, deterministic=False):
        """
        Sample an action from the policy (uses tanh-squashed distribution).
        """
        action_mean, action_std, value = self.forward(state)

        # Create a tanh-squashed distribution
        base_dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(base_dist, TanhTransform())

        if deterministic:
            # For deterministic action use tanh of mean
            action = torch.tanh(action_mean)
        else:
            # Reparameterized sample for gradient flow
            action = dist.rsample()

        return action, value

    def evaluate_actions(self, state, action):
        """
        Evaluate actions for PPO update.
        Returns action log probabilities, state values, and entropy.
        """
        action_mean, action_std, value = self.forward(state)

        base_dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(base_dist, TanhTransform())

        # action is assumed to be in the transformed space (i.e. tanh-squashed)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Entropy of transformed distribution approximated using base distribution entropy
        try:
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        except Exception:
            # Fallback: use base distribution entropy
            entropy = base_dist.entropy().sum(dim=-1, keepdim=True)

        return action_log_probs, value, entropy

    def get_value(self, state):
        """
        Get state value for critic.
        """
        _, _, value = self.forward(state)
        return value
    
    def process_action(self, action):
        """
        Convert network output (already tanh-squashed) to environment action numpy array.
        """
        # action is expected in range [-1, 1]
        action_np = action.cpu().detach().numpy()[0]

        steer = float(action_np[0])
        gas = float((action_np[1] + 1) / 2)  # Scale from [-1,1] to [0,1]
        brake = float((action_np[2] + 1) / 2)  # Scale from [-1,1] to [0,1]

        return np.array([steer, gas, brake], dtype=np.float32)
