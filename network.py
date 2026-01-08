import torch
import numpy as np
from collections import deque


class ClassificationNetwork(torch.nn.Module):
    def __init__(self, stack_k: int = 4):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        
        Args:
            stack_k: Number of frames to stack (default 4)
        """
        super().__init__()

        # Store frame stacking parameter
        self.stack_k = stack_k
        num_classes = 9
        
        # Smoothing state for inference
        self.p_smooth = None
        self.smooth_alpha = 0.35

        # Define the specific architecture here
        # Convolutional layers - accept 3*stack_k channels
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3*self.stack_k, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 96 → 48
            
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 48 → 24
            
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 24 → 12
        )
        
        # Adaptive pooling to reduce spatial size further
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))  # 12 → 6
        
        # Fully connected layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(32 * 6 * 6, 256),  # Correct size after conv + adaptive pooling
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Output layer
        self.output_layer = torch.nn.Linear(128, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3*stack_k)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # Verify stacked frame shape
        assert observation.shape[-1] == 3*self.stack_k, \
            f"Expected {3*self.stack_k} channels, got {observation.shape[-1]}"
        
        # permute input from [batch_size,image_width, image_height, rgb_channels]
        # to [batch_size, channels, image_height, image_width]
        x = observation.permute(0, 3, 1, 2)
        # Normalize input to [0, 1]
        x = x.float() / 255.0
        x = self.conv_layers(x)
        x = self.avgpool(x)  # Apply adaptive pooling
        x = x.reshape(x.size(0), -1)  # Flatten the output

        x = self.fc_layers(x)
        x = self.output_layer(x)
        # x = self.softmax(x)
        return x
        
    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Returns class indices for use with
        CrossEntropyLoss.
        actions:        python list of N torch.Tensors of size 3
        return          torch.Tensor of shape (N,) with dtype torch.long
        """
        """
        Classes:
               0                1               2             3         4     5     6      7      8 
        [left and brake, right and brake, left and gas, right and gas, gas, brake, left, right, nothing]
        """
        class_indices = []

        # Iterate through the list of actions
        for action in actions:
            # Use thresholds instead of exact equality for robustness with noisy expert actions
            steer = action[0]
            gas = action[1]
            brake = action[2]

            # Determine the action class based on the values in the action list
            if steer < -0.1 and brake > 0.1:
                idx = 0  # Left and brake steer class
            elif steer > 0.1 and brake > 0.1: 
                idx = 1  # Right and brake steer class
            elif steer < -0.1 and gas > 0.1:
                idx = 2  # Left and gas steer class
            elif steer > 0.1 and gas > 0.1:
                idx = 3  # Right and gas steer class
            elif gas > 0.1:
                idx = 4  # Gas class
            elif brake > 0.1:
                idx = 5  # Brake class
            elif steer < -0.1:
                idx = 6  # Left class
            elif steer > 0.1:
                idx = 7  # Right class
            else:  # No significant action
                idx = 8  # Do nothing class

            class_indices.append(idx)

        return torch.tensor(class_indices, dtype=torch.long)


    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        Uses probability smoothing to reduce oscillation.
        scores:         torch.Tensor of size (batch_size, number_of_classes) or (number_of_classes,)
        return          (float, float, float)
        """
        # Handle single-sample input
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        """
        Classes:
        [left and brake, right and brake, left and gas, right and gas, gas, brake, left, right, nothing]
        """
        # Compute probabilities
        p = torch.softmax(scores, dim=1)
        
        # Apply exponential smoothing
        if self.p_smooth is None:
            self.p_smooth = p.detach()
        else:
            self.p_smooth = self.smooth_alpha * p.detach() + (1 - self.smooth_alpha) * self.p_smooth
        
        # Choose class from smoothed probabilities
        max_class = torch.argmax(self.p_smooth, dim=1).item()
        
        action = [0.0, 0.0, 0.0]  # Initialize action as [steer, gas, brake]
        if max_class == 0:     # left and brake
            action[0] = -1.0 
            action[2] = 0.8
        elif max_class == 1:   # right and brake 
            action[0] = 1.0
            action[2] = 0.8
        elif max_class == 2:   # left and gas 
            action[0] = -1.0
            action[1] = 0.5
        elif max_class == 3:    # right and gas
            action[0] = 1.0
            action[1] = 0.5
        elif max_class == 4:    # gas (FIXED: was separate 'if', now continuous elif)
            action[1] = 0.5 
        elif max_class == 5:    # brake
            action[2] = 0.8      
        elif max_class == 6:    # left
            action[0] = -1.0
        elif max_class == 7:    # right
            action[0] = 1.0 
        elif max_class == 8:    # nothing
            action = [0.0, 0.0, 0.0]

        # Create commands
        steer = action[0]
        gas = action[1]
        brake = action[2]
        return (steer,gas,brake)

    def reset_policy_state(self):
        """
        Reset the smoothing state. Call this at the beginning of each episode.
        """
        self.p_smooth = None

    # def extract_sensor_values(self, observation, batch_size):
    #     """
    #     observation:    python list of batch_size many torch.Tensors of size
    #                     (96, 96, 3)
    #     batch_size:     int
    #     return          torch.Tensors of size (batch_size, 1),
    #                     torch.Tensors of size (batch_size, 4),
    #                     torch.Tensors of size (batch_size, 1),
    #                     torch.Tensors of size (batch_size, 1)
    #     """
    #     speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
    #     speed = speed_crop.sum(dim=1, keepdim=True) / 255
    #     abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    #     abs_sensors = abs_crop.sum(dim=1) / 255
    #     steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
    #     steering = steer_crop.sum(dim=1, keepdim=True)
    #     gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
    #     gyroscope = gyro_crop.sum(dim=1, keepdim=True)
    #     return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


class FrameStackWrapper:
    """
    Gymnasium-compatible wrapper that stacks the last K frames.
    Returns observations of shape (96, 96, 3*K) with channels concatenated.
    """
    def __init__(self, env, stack_k: int = 4):
        """
        Args:
            env: Gymnasium environment (CarRacing-v3)
            stack_k: Number of frames to stack
        """
        self.env = env
        self.stack_k = stack_k
        self.frame_stack = deque(maxlen=stack_k)
        
    def reset(self, **kwargs):
        """
        Reset the environment and initialize frame stack.
        Returns:
            stacked_obs: np.ndarray of shape (96, 96, 3*stack_k)
            info: dict
        """
        obs, info = self.env.reset(**kwargs)
        
        # Fill the deque with K copies of the initial frame
        for _ in range(self.stack_k):
            self.frame_stack.append(obs)
        
        # Concatenate along channel axis
        stacked_obs = np.concatenate(list(self.frame_stack), axis=2)
        return stacked_obs, info
    
    def step(self, action):
        """
        Take a step in the environment.
        Returns:
            stacked_obs: np.ndarray of shape (96, 96, 3*stack_k)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Append new frame (oldest is automatically removed by deque)
        self.frame_stack.append(obs)
        
        # Concatenate along channel axis
        stacked_obs = np.concatenate(list(self.frame_stack), axis=2)
        return stacked_obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped environment."""
        return getattr(self.env, name)


def test_model():
    """Test the ClassificationNetwork with frame stacking."""
    print("Testing ClassificationNetwork with frame stacking...")
    
    # Create model with stack_k=4
    model = ClassificationNetwork(stack_k=4)
    model.eval()
    
    # Create dummy input: batch of 2, 96x96, 12 channels (4 frames * 3 RGB)
    dummy_input = torch.zeros((2, 96, 96, 12), dtype=torch.uint8)
    
    # Run forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (2, 9), f"Expected shape (2, 9), got {output.shape}"
    print(f"✓ Model output shape: {output.shape}")
    
    # Test scores_to_action
    action = model.scores_to_action(output[0])
    assert len(action) == 3, f"Expected 3 action values, got {len(action)}"
    print(f"✓ Action output: {action}")
    
    # Test reset_policy_state
    model.reset_policy_state()
    assert model.p_smooth is None, "p_smooth should be None after reset"
    print("✓ reset_policy_state() works")
    
    # Test actions_to_classes
    actions = [torch.tensor([0.0, 0.5, 0.0]), torch.tensor([-1.0, 0.0, 0.8])]
    class_indices = model.actions_to_classes(actions)
    assert class_indices.shape == (2,), f"Expected shape (2,), got {class_indices.shape}"
    assert class_indices.dtype == torch.long, f"Expected dtype torch.long, got {class_indices.dtype}"
    print(f"✓ actions_to_classes output: {class_indices}")
    
    print("✓ All model tests passed!\n")


def test_wrapper():
    """Test the FrameStackWrapper (mock environment)."""
    print("Testing FrameStackWrapper...")
    
    # Create a mock environment
    class MockEnv:
        def reset(self, **kwargs):
            obs = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            return obs, {}
        
        def step(self, action):
            obs = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            return obs, 0.0, False, False, {}
    
    # Wrap the mock environment
    env = FrameStackWrapper(MockEnv(), stack_k=4)
    
    # Test reset
    stacked_obs, info = env.reset()
    assert stacked_obs.shape == (96, 96, 12), f"Expected shape (96, 96, 12), got {stacked_obs.shape}"
    print(f"✓ Reset returns shape: {stacked_obs.shape}")
    
    # Test step
    stacked_obs, reward, terminated, truncated, info = env.step((0.0, 0.5, 0.0))
    assert stacked_obs.shape == (96, 96, 12), f"Expected shape (96, 96, 12), got {stacked_obs.shape}"
    print(f"✓ Step returns shape: {stacked_obs.shape}")
    
    print("✓ All wrapper tests passed!\n")


if __name__ == "__main__":
    test_model()
    test_wrapper()
    print("=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)
