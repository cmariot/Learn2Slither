import random
import torch
from model import DeepQNetwork, QTrainer
import os
import pickle
from Score import Score
from numpy import ndarray
import math
from ReplayMemory import ReplayMemory


MAX_MEMORY = 200_000
BATCH_SIZE = 10_000

LEARNING_RATE = 0.001

EPSILON_START = 1
EPSILON_END = 0.005
EPSILON_DECAY = 100


class Agent:

    def __init__(self, args: tuple):

        self.lr = LEARNING_RATE
        self.epsilon = EPSILON_START
        self.gamma = 0.8

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
        self.epsilon = (EPSILON_START - EPSILON_END) * \
            math.exp(-1. * episode / EPSILON_DECAY)

    def choose_action(self, state, nb_games):

        if not self._train:
            self.epsilon = 0
        else:
            self.epsilon_decay(nb_games)

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        # Exploration vs exploitation
        if random.uniform(0, 1) < self.epsilon:
            self.choice_type = "exploration"
            self.action = random.randint(0, 3)
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

        if not self._train:
            return

        self.memory.push(state, action, reward, next_state, is_alive)
        self.trainer.train_step(state, action, reward, next_state, is_alive)

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
