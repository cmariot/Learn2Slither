import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork(nn.Module):

    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(144, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.layers(x)


class QTrainer:

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

        prediction = self.model(state)
        target = prediction.clone()
        target = target.detach()

        if game_over:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * action

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
        return loss.item()
