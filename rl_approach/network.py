import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from typing import Tuple

#           ┌──────────────┐
#           │ Shared CNN+FC│  ← receives gradients from both losses
#           └──────┬───────┘
#       ┌──────────┴──────────┐
#       │                     │
#  Actor head             Critic head
# (policy loss)           (value loss)
# (entropy loss)


class ActorCritic(nn.Module):
    """ 
    PPO can be interpreted as an actor-critic algorithm in that it has
    a policy (actor) and a value function (critic).
    So, this class produces two outputs from one shared neural network backbone:
        - Actor (the policy) outputs the mean and log-standard-deviation parameters
          of a Gaussian distribution over continuous actions (steer, gas, brake).
        - Critic (the value function) outputs state value, V(s), and is used
          for advantage estimation.
            - We need the critic because the environment just reads the reward for that single observation,
              whereas the value function estimates total rewards of the policy, including in the future.
              So the critic learns to estimate the value of being in a given state under the current policy (the value does indeed depend on the policy)
              - Target: return_t​=rt​+γrt+1​+γ2rt+2​+…
              - Critic prediction: V_θ​(s_t​
              - Critic loss = ||V_θ​(s_t​)-return_t||^2
            - This led me to the question: the value function depends on the policy, but the value head and the policy head are separate,
              so how do you estimate the value without knowing the policy parameters?
              - Answer:
                - The critic never actually needs to know the policy parameters directly.
                - Since both heads share the same CNN and FC layers, the features extracted are influenced by both the policy
                  and value losses during training.
            - And then another question: the policy is changing during training, does that mean the value function is always "chasing a moving target"?
               - Answer:
                - Yes, it does. This is a known challenge in actor-critic methods, and techniques like using target networks or
                  slower updates for the critic are often employed to stabilize training.
                - Policy updates are slow and we make sure of it using clipping
                    The Clip Function: The probability ratio (probability of doing action given state with new to old policy)
                    is clipped to a small interval around 1, typically [1-ϵ ,1+ϵ ], where ϵ is a hyperparameter, often set to 0.2.
        - Sharing a backbone between the actor and critic has several benefits:
            - Efficiency: Fewer parameters to learn, faster inference.
            - Regularization: Shared features can help prevent overfitting - they have different goals in a way that complements each other.
            - Coordinated learning: Updates to one head can positively influence the other.


    ActorCritic inherits from nn.Module which gives:
        - Parameter management (.parameters(), .named_parameters())
        - Model saving and loading (.state_dict(), .load_state_dict())
        - Device management (to(), cuda(), cpu())
        - Gradient computation and backpropagation (zero_grad(), backward(), step())
    """
    def __init__(self, num_inputs: int = 4, num_actions: int = 3) -> None:
        """
        Initialize the Actor-Critic network.

        Args:
            num_inputs: Number of input channels. Default is 4 for stacking 4 consecutive 
                       grayscale frames, which allows the CNN to perceive short-term motion 
                       and velocity. This approach follows Atari DQN and other successful RL 
                       implementations.
            num_actions: Number of continuous action outputs. Default is 3 for CarRacing-v3: 
                        [steer, gas, brake], each in range [-1, 1].
        """
        super(ActorCritic, self).__init__()
        
        # Shared CNN feature extractor
        # First layer has large kernel and stride to downsample quickly because the input is 4 frames of 84x84.
        # Kernel size of 8 extracts coarse spatial patterns and is a good size for seeing the grass vs track, big curves
        # Outputting channel of 32 is a good balance between expressiveness and computational cost, "enough to get basic visual primitives"
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        # Second layer captures slightly smaller features like road edges and car boundaries
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Third layer captures fine details using smaller kernel and just a stride of 1 so as not to lose spatial resolution
        # Spatial resolution has already been shrunk a lot so we can afford a small kernel here
        # The final conv layer is the "feature blender" that combines all extracted features before feeding the fully connected layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Summary:
        #   conv1: massive downsampling + big visual primitives
        #   conv2: medium receptive features + more downsampling
        #   conv3: fine feature extraction + no downsampling
        
        # Calculate conv output size
        def conv2d_size_out(size: int, kernel_size: int, stride: int) -> int:
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
        # action_log_std is stored as a free parameter, pytorch registers it as a learnable parameter, initialized to zeros,
        # representing log standard deviation of actions.
        # while not connected to the actor MLP it influences the action distribution, which affects the log probability,
        # which then affects the PPO loss.
        
        # Critic head (value function)
        self.fc_critic = nn.Linear(512, 256)
        self.value = nn.Linear(256, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize action mean with smaller values
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input state tensor of shape (batch_size, num_inputs, 84, 84).

        Returns:
            action_mean: Mean of action distribution, shape (batch_size, num_actions).
            action_std: Standard deviation of action distribution, shape (batch_size, num_actions).
            value: State value estimate, shape (batch_size, 1).
        """
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
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy using a tanh-squashed Gaussian distribution.

        Args:
            state: Input state tensor of shape (batch_size, num_inputs, 84, 84).
            deterministic: If True, return deterministic action (tanh of mean). 
                          If False, sample from the distribution.

        Returns:
            action: Sampled or deterministic action, shape (batch_size, num_actions), range [-1, 1].
            value: State value estimate, shape (batch_size, 1).
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

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            state: Input state tensor of shape (batch_size, num_inputs, 84, 84).
            action: Action tensor to evaluate, shape (batch_size, num_actions), range [-1, 1].

        Returns:
            action_log_probs: Log probabilities of the actions, shape (batch_size, 1).
            value: State value estimates, shape (batch_size, 1).
            entropy: Entropy of the action distribution, shape (batch_size, 1).
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

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimate from the critic.

        Args:
            state: Input state tensor of shape (batch_size, num_inputs, 84, 84).

        Returns:
            value: State value estimate, shape (batch_size, 1).
        """
        _, _, value = self.forward(state)
        return value
    
    def process_action(self, action: torch.Tensor) -> np.ndarray:
        """
        Convert network output action to environment-compatible numpy array.

        Args:
            action: Action tensor from the network, shape (1, num_actions), range [-1, 1].

        Returns:
            Numpy array of shape (3,) with [steer, gas, brake].
            - steer: float in range [-1, 1]
            - gas: float in range [0, 1]
            - brake: float in range [0, 1]
        """
        # action is expected in range [-1, 1]
        action_np = action.cpu().detach().numpy()[0]

        steer = float(action_np[0])
        gas = float((action_np[1] + 1) / 2)  # Scale from [-1,1] to [0,1]
        brake = float((action_np[2] + 1) / 2)  # Scale from [-1,1] to [0,1]

        return np.array([steer, gas, brake], dtype=np.float32)
