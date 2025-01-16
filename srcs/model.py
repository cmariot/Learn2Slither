import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 256)
        # self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(256, 4)
        self.load()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self, file_name)
        print("Model saved")

    def load(self, file_name="model.pth"):
        file_name = os.path.join("./model", file_name)
        if os.path.exists(file_name):
            self = torch.load(file_name)
            print("Model loaded")
        return self


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
