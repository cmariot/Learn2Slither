import random
from collections import deque
import torch
from model import Linear_QNet, QTrainer
import os
import pickle
from Score import Score
from numpy import ndarray


MAX_MEMORY = 100_000
BATCH_SIZE = 100_000


class Agent:

    def __init__(self, args: tuple):

        self.lr = 0.001
        self.epsilon = 0.01
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet()
        self.trainer = QTrainer(self.model, self.lr, self.gamma)

        if args.new_model is False:
            if args.model_path:
                self.load_model(args.model_path)
            else:
                self.load_max_trained_model()

        self._train = args.train
        self.dont_save = args.dont_save

    def choose_action(self, state, nb_games):

        # Espilon-greedy strategy
        # Epsilon is the probability of choosing a random action
        # 1 - epsilon is the probability of choosing the best action

        # The epsilon decreases as the number of games increases
        # self.epsilon = 0.01 + 0.2 * (1 - min(nb_games, 1000) / 1000)

        def update_epsilon(nb_games):
            if nb_games < 10:
                return 0.5
            elif nb_games < 100:
                return 0.2
            elif nb_games < 300:
                return 0.05
            elif nb_games < 500:
                return 0.025
            elif nb_games < 1000:
                return 0.01
            else:
                return 0

        if not self._train:
            self.epsilon = 0
        else:
            self.epsilon = 0.01

        n = random.uniform(0, 1)

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        # Exploration vs exploitation
        if n < self.epsilon:
            # Random action : action = random.randint(0, 3)
            # vs. Second best action :
            return torch.argsort(prediction, descending=True)[1].item()

        return torch.argmax(prediction).item()

    def train(
                self,
                state: ndarray,
                action: int,
                reward: int,
                next_state: ndarray,
                is_alive: bool
            ):
        self.train_short_memory(state, action, reward, next_state, is_alive)
        self.learn(state, action, reward, next_state, is_alive)

    def learn(
                self,
                state: ndarray,
                action: int,
                reward: int,
                next_state: ndarray,
                is_alive: bool
            ):

        if not self._train:
            return

        self.memory.append((state, action, reward, next_state, is_alive))
        if len(self.memory) > MAX_MEMORY:
            self.memory.popleft()

    def train_long_memory(self):

        if not self._train:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, is_alive in mini_sample:
            self.trainer.train_step(
                state, action, reward, next_state, is_alive
            )

    def train_short_memory(
                self,
                state: ndarray,
                action: int,
                reward: int,
                next_state: ndarray,
                is_alive: bool
            ):

        if not self._train:
            return

        self.trainer.train_step(state, action, reward, next_state, is_alive)

    def save(self, scores: Score):

        if self.dont_save or not self._train:
            return

        model_path = os.path.join(os.getcwd(), "models")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Create a directory to store the model
        model_directory_name = f"game_{scores.game_number}"

        model_path = os.path.join(model_path, model_directory_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_file = os.path.join(model_path, "model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved as {model_file}")

        scores.save(model_path)

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

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model path '{model_path}' does not exist."
            )
        model_file = os.path.join(model_path, "model.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file '{model_file}' does not exist."
            )
        with open(model_file, "rb") as f:
            agent = pickle.load(f)
            # Check if the model is an instance of the Agent class
            if not isinstance(agent, Agent):
                raise TypeError(
                    f"Model file '{model_file}' is not an instance of the" +
                    " Agent class."
                )
            for attr in agent.__dict__:
                setattr(self, attr, getattr(agent, attr))
            print(f"Model loaded from {model_file}")
