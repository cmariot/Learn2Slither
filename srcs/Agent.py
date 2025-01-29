import random
import torch
from model import DeepQNetwork, QTrainer
import os
import pickle
from Score import Score
from numpy import ndarray
import math
from ReplayMemory import ReplayMemory
import numpy as np


MAX_MEMORY = 500_000
BATCH_SIZE = 1_024

LEARNING_RATE = 10e-4
GAMMA = 0.99

EPSILON_START = 0.1
EPSILON_END = 0.001
EPSILON_DECAY = 1000


# Random seed
# random.seed(4242)


class Agent:

    def __init__(self, args: tuple):

        self.lr = LEARNING_RATE
        self.epsilon = EPSILON_START
        self.gamma = GAMMA

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

    def epsilon_decay(self, nb_games):

        if EPSILON_DECAY == 0:
            self.epsilon = 0.0
            return

        self.epsilon = (EPSILON_START - EPSILON_END) * \
            math.exp(- nb_games / EPSILON_DECAY)

        self.epsilon = min(EPSILON_START, max(EPSILON_END, self.epsilon))

    def choose_action(self, state, nb_games):

        self.epsilon_decay(nb_games)

        prediction = self.model(torch.tensor(state, dtype=torch.float))

        # Exploration vs exploitation
        if random.uniform(0, 1) < self.epsilon:
            self.choice_type = "exploration"
            choice_index = random.randint(1, 3)
            self.action = \
                torch.argsort(prediction, descending=True)[choice_index].item()
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
        game_over: bool
    ):
        """
        Trains the model on a single experience and adds it to memory.
        """
        if not self._train:
            return

        # Entraînement sur une seule expérience
        self.trainer.train_step(
            torch.tensor(state, dtype=torch.float).unsqueeze(0),
            torch.tensor([action], dtype=torch.long),
            torch.tensor([reward], dtype=torch.float),
            torch.tensor(next_state, dtype=torch.float).unsqueeze(0),
            torch.tensor([game_over], dtype=torch.bool)
        )

        # Ajout de l'expérience dans la mémoire
        self.memory.push((state, action, reward, next_state, game_over))

    def train_long_memory(self):

        """
        Experience replay mechanism: trains the model on a batch of past
        experiences.
        """

        if not self._train:
            return

        # Échantillonner un batch aléatoire
        memory_len = len(self.memory)
        batch_size = min(memory_len, BATCH_SIZE)

        # Échantillonner un batch aléatoire
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, game_over = zip(*batch)

        states, actions, rewards, next_states, game_over = zip(*batch)

        # Entraîner le modèle sur le batch
        self.trainer.train_step(
            torch.tensor(np.array(states), dtype=torch.float),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float),
            torch.tensor(np.array(next_states), dtype=torch.float),
            torch.tensor(np.array(game_over), dtype=torch.bool)
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
                to_remove = []
                for i in range(len(model_directory)):
                    # Check if the directory name is in the right format
                    model_dir_array = model_directory[i].split("_")
                    if len(model_dir_array) != 2:
                        to_remove.append(i)
                        continue
                    model_directory[i] = int(model_dir_array[1])
                for i in to_remove:
                    model_directory.pop(i)
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
