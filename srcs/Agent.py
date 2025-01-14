import random
from collections import deque
import torch
from model import Linear_QNet, QTrainer
import os
import pickle


MAX_MEMORY = 1_000_000
BATCH_SIZE = 10_000
LEARNING_RATE = 0.001


class Agent:

    def __init__(self, model_path=None):
        self.epsilon = 0.01
        self.gamma = 0.9
        self.lr = LEARNING_RATE
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(16, 256, 4)
        self.trainer = QTrainer(self.model, self.lr, self.gamma)

        if not model_path:
            self.load_max_trained_model()

    def load_max_trained_model(self):
        # Load the most trained model if it exists
        model_path = os.path.join(os.getcwd(), "models")
        if os.path.exists(model_path):
            model_directory = os.listdir(model_path)
            if model_directory:
                for i in range(len(model_directory)):
                    model_directory[i] = int(model_directory[i].split("_")[1])
                model_directory.sort()
                model_directory = f"game_{model_directory[-1]}"
                model_path = os.path.join(model_path, model_directory)
                model_file = os.path.join(model_path, "model.pkl")
                if os.path.exists(model_file):
                    with open(model_file, "rb") as f:
                        agent = pickle.load(f)
                        for attr in agent.__dict__:
                            setattr(self, attr, getattr(agent, attr))
                        print(f"Model loaded from {model_file}")

    def choose_action(self, state, nb_games):
        # Exploration vs exploitation

        # Espilon-greedy strategy
        # Epsilon is the probability of choosing a random action
        # 1 - epsilon is the probability of choosing the best action

        # The epsilon decreases as the number of games increases
        # self.epsilon = 0.01 + 0.2 * (1 - min(nb_games, 1000) / 1000)
        self.epsilon = 0.01

        # print(f"epsilon: {self.epsilon}")

        n = random.uniform(0, 1)

        if n < self.epsilon:
            # Random action
            # action = random.randint(0, 3)
            # vs
            # Second best action
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argsort(prediction, descending=True)[1].item()
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()

        return action

    def learn(
            self,
            state,
            action,
            reward,
            next_state,
            snake_alive,
            score_evolution
            ):

        self.memory.append((state, action, reward, next_state, snake_alive))
        if len(self.memory) > MAX_MEMORY:
            self.memory.popleft()

        # Update the score evolution
        score_evolution.score += reward
        score_evolution.turn += 1

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, snake_alive in mini_sample:
            self.trainer.train_step(
                state, action, reward, next_state, snake_alive
            )

    def train_short_memory(self, state, action, reward, next_state, snake_alive):
        self.trainer.train_step(state, action, reward, next_state, snake_alive)

    def save(self, agent, score_evolution):

        model_path = os.path.join(os.getcwd(), "models")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Create a directory to store the model
        model_directory_name = f"game_{score_evolution.game_number}"

        model_path = os.path.join(model_path, model_directory_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_file = os.path.join(model_path, "model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(agent, f)
        print(f"Model saved as {model_file}")

        score_evolution.save(model_path)
        del score_evolution
