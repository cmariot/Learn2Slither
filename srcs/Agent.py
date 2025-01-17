import random
from collections import namedtuple, deque
import torch
from model import DeepQNetwork, QTrainer
import os
import pickle
from Score import Score
from numpy import ndarray
import math


MAX_MEMORY = 200_000
BATCH_SIZE = 10_000

Transition = namedtuple(
    # a named tuple representing a single transition in our environment.
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'game_over')
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:

    def __init__(self, args: tuple):

        self.lr = 0.001
        self.epsilon = 0.01
        self.gamma = 0.99

        self.memory = ReplayMemory(MAX_MEMORY)
        self.model = DeepQNetwork()
        self.trainer = QTrainer(self.model, self.lr, self.gamma)

        if args.new_model is False:
            if args.model_path:
                self.load_model(args.model_path)
            else:
                self.load_max_trained_model()

        self._train = args.train
        self.dont_save = args.dont_save

    def epsilon_decay(self, episode):
        # Logarithmic decay
        self.epsilon = 0.01 + (1 - 0.01) * (1 - math.log10((episode + 1) / 25))
        self.epsilon = min(1, max(0.01, self.epsilon))

    def choose_action(self, state, nb_games):

        # Espilon-greedy strategy
        # Epsilon is the probability of choosing a random action
        # 1 - epsilon is the probability of choosing the best action

        # The epsilon decreases as the number of games increases
        if not self._train:
            self.epsilon = 0
        else:
            self.epsilon_decay(nb_games)

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        # Exploration vs exploitation

        if random.uniform(0, 1) < self.epsilon:

            self.choice_type = "exploration"

            if self.epsilon > 0.01:
                # Random action : action = random.randint(0, 3)
                self.action = random.randint(0, 3)
                return self.action

            # vs. Second best action :
            self.action = torch.argsort(prediction, descending=True)[1].item()
            return self.action

        self.choice_type = "exploitation"
        self.action = torch.argmax(prediction).item()
        return self.action

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

        self.memory.push(state, action, reward, next_state, is_alive)
        if len(self.memory) > MAX_MEMORY:
            self.memory.popleft()

    def train_long_memory(self):
        """
        experience replay mechanism
        """

        if not self._train:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = self.memory.sample(BATCH_SIZE)
        else:
            mini_sample = self.memory.sample(len(self.memory))

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

        """
        Save the agent and the scores at the end of the training
        """

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
