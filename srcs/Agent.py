import random
from collections import deque
import torch
from model import Linear_QNet, QTrainer
import numpy as np
from constants import RED, BLUE, RESET

MAX_MEMORY = 1_000_000
BATCH_SIZE = 1_000


class Agent:

    def __init__(self):
        self.nb_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.lr = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(20, 256, 4)
        self.trainer = QTrainer(self.model, self.lr, self.gamma)

    def choose_action(self, state, nb_games):

        # Logarithmic decay of epsilon
        self.nb_games = nb_games

        print(f"Number of games: {self.nb_games}")

        if nb_games % 100 == 0:
            self.epsilon += np.log(nb_games + 1) / 100
            self.epsilon = min(0.999, self.epsilon)

        print(f"Epsilon: {self.epsilon}")

        if random.uniform(0, 1) > self.epsilon:
            print(f"{RED}Exploring{RESET}")
            actions = ("up", "down", "left", "right")
            return random.choice(actions)
        else:
            print(f"{BLUE}Exploiting{RESET}")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            return ["up", "down", "left", "right"][move]

    def get_reward(self, state, action, snake_alive):
        directions = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }
        idx = list(directions.keys()).index(action)
        idxs = [value + idx for value in range(0, 20, 4)]

        (
            green_apple_distance,
            red_apple_distance,
            wall_distance,
            body_distance,
            empty_distance
        ) = state[idxs]

        reward = 0
        if red_apple_distance == 1 and snake_alive:
            reward = -5
        elif green_apple_distance == 1:
            reward = 5
        elif empty_distance == 1:
            reward = -1
        elif (
            red_apple_distance == 1 and not snake_alive or
            wall_distance == 1 or
            body_distance == 1
        ):
            reward = -30

        return reward

    def learn(self, state, action, reward, next_state, snake_alive):
        self.memory.append((state, action, reward, next_state, snake_alive))
        if len(self.memory) > MAX_MEMORY:
            self.memory.popleft()

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, snake_alive = zip(*mini_sample)
        self.trainer.train_step(
            states, actions, rewards, next_states, snake_alive
        )

    def train_short_memory(self, state, action, reward, next_state, snake_alive):
        self.trainer.train_step(state, action, reward, next_state, snake_alive)
