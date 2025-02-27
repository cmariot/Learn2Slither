import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# torch.manual_seed(0)


class DeepQNetwork(nn.Module):

    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.input_layer = nn.Linear(16, 256)
        self.hidden_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 4)

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
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        game_overs: torch.Tensor
    ):
        """
        Performs a training step on a batch or a single experience.
        """

        # Prediction of the Q values based on the current state
        predictions: torch.Tensor = self.model(states)

        # Get the Q value of the action taken
        q_value = \
            predictions.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():

            # Prediction of the Q values for the next state
            next_predictions = self.model(next_states)

            # Get the maximum Q value for the next state
            next_action = torch.max(next_predictions, dim=1)[0]

            # Compute the target Q values
            new_q_value = \
                rewards + (self.gamma * next_action * (~game_overs))

        # Update the model
        self.optimizer.zero_grad()
        loss = self.loss.forward(q_value, new_q_value)
        loss.backward()
        self.optimizer.step()
