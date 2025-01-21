import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# PyTorch seed
torch.manual_seed(0)


class DeepQNetwork(nn.Module):

    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.input_layer = nn.Linear(16, 256)
        self.output_layer = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x


class QTrainer:

    def __init__(self, model: DeepQNetwork, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):

        if len(state.shape) == 1:
            # Ajout de dimension : Utilisation de unsqueeze(0) pour ajouter une
            # dimension supplémentaire, car les méthodes de PyTorch s'attendent
            # souvent à des lots de données (batch).
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

        prediction: torch.Tensor = self.model(state)

        target = prediction.clone().detach()
        if game_over:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * action

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
        return loss.item()
