from collections import deque
import random


# random.seed(0)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, tuple):
        self.memory.append(tuple)

    def sample(self, batch_size) -> list[tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
