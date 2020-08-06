import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        h = 18  # hidden dimension
        super(DQN, self).__init__()
        self.neuralnet = nn.Sequential(
            nn.Linear(self.state_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, self.action_size))

    def forward(self, x):
        return self.neuralnet(x)


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        learning_rate = 0.0025
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)

    def backward(self, y, y_target):
        loss = self.loss_fn(y, y_target)

        # Zero-out all the gradients
        self.optimizer.zero_grad()

        # Backward pass: compute gradient of the loss
        loss.backward()

        # Calling the step function on an Optimizer in order to apply the gradients (update)
        self.optimizer.step()

