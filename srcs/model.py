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
        # self.hidden_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        # x = F.relu(self.hidden_layer(x))
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
        state: np.ndarray | tuple[np.ndarray],
        action: int | tuple[int],
        reward: float | tuple[float],
        next_state: np.ndarray | tuple[np.ndarray],
        game_over: bool | tuple[bool]
    ):

        state: torch.Tensor = torch.tensor(state, dtype=torch.float)
        action: torch.Tensor = torch.tensor(action, dtype=torch.int)
        reward: torch.Tensor = torch.tensor(reward, dtype=torch.float)
        next_state: torch.Tensor = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )

        # Prediction of the Q values based on the state
        prediction: torch.Tensor = self.model(state)

        # Update the Q value for the action taken
        optimized_prediction = prediction.clone()
        for i in range(len(game_over)):
            new_Q = reward[i]
            if not game_over[i]:
                new_Q += self.gamma * torch.max(self.model(next_state[i]))
            optimized_prediction[i][action[i]] = new_Q

        # Update the model
        self.optimizer.zero_grad()
        self.loss(optimized_prediction, prediction).backward()
        self.optimizer.step()

    # def train_step(
    #         self,
    #         state: torch.Tensor,
    #         action: torch.Tensor,
    #         reward: torch.Tensor,
    #         next_state: torch.Tensor,
    #         game_over: torch.Tensor
    # ) -> float:

    #     """
    #     This method is called at each turn of the game to train the agent.
    #     It's also called at the end of the game to train the agent on the
    #     long memory.
    #     """

    #     # print("Train step")
    #     # print("State: ", state.shape)
    #     # print("Action: ", action.shape)
    #     # print("Reward: ", reward.shape)
    #     # print("Next state: ", next_state.shape)
    #     # print("Game over: ", game_over.shape)

    #     if len(state.shape) == 1:

    #         Ajout de dimension : Utilisation de unsqueeze(0) pour ajouter une
    #         dimension supplémentaire, car les méthodes de PyTorch s'attendent
    #         # souvent à des lots de données (batch).

    #         # print("Adding dimension")

    #         state = torch.unsqueeze(state, 0)
    #         next_state = torch.unsqueeze(next_state, 0)
    #         action = torch.unsqueeze(action, 0)
    #         reward = torch.unsqueeze(reward, 0)
    #         game_over = torch.unsqueeze(game_over, 0)

    #         # print("NEW State: ", state)
    #         # print("NEW Action: ", action)
    #         # print("NEW Reward: ", reward)
    #         # print("NEW Next state: ", next_state)
    #         # print("NEW Game over: ", game_over)

    #     # Prediction of the Q values based on the state
    #     prediction: torch.Tensor = self.model(state)

    #     target = prediction.clone().detach()

    #     # print("Prediction: ", prediction)
    #     # print("Target: ", target)

    #     # Pas sur du fonctionnement dans le cas de la train_long_memory,
    #     # a revoir

    #     batch_size = len(target)
    #     for idx in range(batch_size):
    #         # state_value = state[idx]
    #         action_value = action[idx].item()
    #         reward_value = reward[idx].item()
    #         next_state_value = next_state[idx]
    #         game_over_value = game_over[idx].item()

    #         # print("State value: ", state_value)
    #         # print("Action value: ", action_value)
    #         # print("Reward value: ", reward_value)
    #         # print("Next state value: ", next_state_value)
    #         # print("Game over value: ", game_over_value)

    #         if game_over_value:
    #             target[idx][action_value] = reward_value
    #         else:
    #             target[idx][action_value] = reward_value + self.gamma * \
    #                 torch.max(self.model(next_state_value))

    #     self.optimizer.zero_grad()
    #     loss = self.criterion(target, prediction)
    #     loss.backward()
    #     self.optimizer.step()

    #     # print("")

    #     return loss.item()
