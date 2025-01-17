import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DeepQNetwork(nn.Module):

    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.input_layer = nn.Linear(160, 512)
        self.hidden_layer_1 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 4)

    def forward(self, x):

        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = self.output_layer(x)

        return x


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
