import torch


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')

        # input_channels = 3
        num_classes = 9

        # Define the specific architecture here
        # Convolutional layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )
        
        # Fully connected layers
        self.fc_layers = torch.nn.Sequential(
            # torch.nn.Linear(64*128*12*12, 256),
            torch.nn.Linear(64*32*12*12, 256),
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
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # permute input from [batch_size,image_width, image_height, rgb_channels]
        # to [batch_size, channels, image_height, image_width]
        x = observation.permute(0, 3, 1, 2) 
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)  # Flatten the output

        x = self.fc_layers(x)
        x = self.output_layer(x)
        # x = self.softmax(x)
        return x
        
    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        # Create an empty list to store the class representations for each action
        number_of_classes = 9
        """
        Classes:
               0                1               2             3         4     5     6      7      8 
        [left and brake, right and brake, left and gas, right and gas, gas, brake, left, right, nothing]
        """
        class_representations = []

        # Iterate through the list of actions
        for action in actions:
            # Initialize a tensor of zeros with a size of number_of_classes
            class_tensor = torch.zeros(number_of_classes)

            # Determine the action class based on the values in the action list
            if action[0] == -1.0 and action[2]>0:
                class_tensor[0] = 1  # Left and brake steer class
            elif action[0] == 1.0 and action[2]>0: 
                class_tensor[1] = 1  # Right and brake steer class
            elif action[0] == -1.0 and action[1]>0:
                class_tensor[2] = 1  # Left and gas steer class
            elif action[0] == 1.0 and action[1]>0:
                class_tensor[3] = 1  # Right and gas steer class
            elif action[1] > 0:
                class_tensor[4] = 1  # Gas class
            elif action[2] > 0:
                class_tensor[5] = 1  # Brake class
            elif action[0] == -1.0:
                class_tensor[6] = 1  # Left class
            elif action[0] == 1.0:
                class_tensor[7] = 1  # Right class
            elif action[0] == 0 and action[1] == 0 and action[2] == 0:
                class_tensor[8] = 1  # Do nothing class

            # Append the one-hot encoded class representation to the list
            class_representations.append(class_tensor)

        return class_representations


    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """


        """
        Classes:
        [left and brake, right and brake, left and gas, right and gas, gas, brake, left, right, nothing]
        """
        # Map the predicted class back to the corresponding action
        softmaxed_score = self.softmax(scores)
        max_class = torch.argmax(softmaxed_score)
        action = [0.0, 0.0, 0.0]  # Initialize action as [steer, gas, brake]
        if max_class == 0:     # left and brake
            action[0] = -1.0 
            action[2] = 0.8
        elif max_class == 1:   # right and brake 
            action[1] = 1.0
            action[2] = 0.8
        elif max_class == 2:   # left and gas 
            action[0] = -1.0
            action[1] = 0.5
        elif max_class == 3:    # right and brake
            action[0] = 1.0
            action[1] = 0.5
        if max_class == 4:      # gas
            action[1] = 0.5 
        elif max_class == 5:    # brake
            action[2] = 0.8      
        elif max_class == 6:    # left
            action[0] = -1.0
        elif max_class == 7:    # right
            action[0] = 1.0 
        elif max_class == 8:    # nothing - not necessary because action list is not altered by this elif, but just to check  
            action = [0.0, 0.0, 0.0]

        # Create commands
        steer = action[0]
        gas = action[1]
        brake = action[2]
        return (steer,gas,brake)

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
