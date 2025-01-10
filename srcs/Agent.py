import random
from collections import deque
import torch
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 0.001


class Agent:

    def __init__(self):

        self.nb_games = 0

        # Epsilon : Control the randomness between exploration and exploitation
        self.epsilon = 0

        self.gamma = 0
        self.lr = LEARNING_RATE
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(12, 256, 4)
        self.trainer = QTrainer(self.model, self.lr, self.gamma)

        self.score = 0
        self.high_score = 0
        self.current_max_score = 0

    def choose_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        print("Move: ", move)
        return move
        # return ["up", "down", "left", "right"][move]

    def learn(self, state, action, reward, next_state, snake_alive):
        self.memory.append((state, action, reward, next_state, snake_alive))
        if len(self.memory) > MAX_MEMORY:
            self.memory.popleft()
        self.score += reward

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, snake_alive in mini_sample:
            print("\nTraining Long Memory")
            print("State: ", state)
            print("Action: ", action)
            print("Reward: ", reward)
            print("Next State: ", next_state)
            print("Snake Alive: ", snake_alive)
        #     self.trainer.train_step(
        #         state, action, reward, next_state, snake_alive
        #     )

    def train_short_memory(self, state, action, reward, next_state, snake_alive):
        print("\nTraining Short Memory")
        print("State: ", state)
        print("Action: ", action)
        print("Reward: ", reward)
        print("Next State: ", next_state)
        print("Snake Alive: ", snake_alive)

        self.trainer.train_step(state, action, reward, next_state, snake_alive)
