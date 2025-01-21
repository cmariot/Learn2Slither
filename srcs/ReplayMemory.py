from collections import namedtuple, deque
import random


random.seed(0)

Transition = namedtuple(
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
