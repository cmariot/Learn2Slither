import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# PyTorch seed
torch.manual_seed(0)


class DeepQNetwork(nn.Module):

    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.input_layer = nn.Linear(16, 256)
        self.hidden_layer = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


class QTrainer:

    def __init__(self, model: DeepQNetwork, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        game_overs: np.ndarray
    ):

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        game_overs = torch.tensor(game_overs, dtype=torch.bool)

        # Prediction of the Q values based on the current state
        predictions = self.model(states)

        # Prediction of the Q values for the next state
        next_predictions = self.model(next_states)

        with torch.no_grad():
            # Get the maximum Q value for the next state
            max_next_q_values = torch.max(next_predictions, dim=1)[0]
            target_q_values = \
                rewards + (self.gamma * max_next_q_values * (~game_overs))

        # Gather the Q values corresponding to the actions taken
        action_q_values = \
            predictions.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Update the model
        self.optimizer.zero_grad()
        loss = self.loss.forward(action_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
