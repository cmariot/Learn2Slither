from collections import namedtuple, deque
import random


random.seed(0)

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ########################### #
# ********** TESTS ********** #
# ########################### #


# def test_push_and_len():
#     capacity = 10
#     memory = ReplayMemory(capacity)
#     assert len(memory) == 0
#     memory.push(1, 2, 3, 4, False)
#     assert len(memory) == 1
#     for _ in range(9):
#         memory.push(1, 2, 3, 4, False)
#     assert len(memory) == 10
#     memory.push(1, 2, 3, 4, False)
#     assert len(memory) == 10


# def test_sample():
#     capacity = 10
#     memory = ReplayMemory(capacity)
#     for i in range(10):
#         memory.push(i, i+1, i+2, i+3, False)
#     sample = memory.sample(5)
#     assert len(sample) == 5
#     assert all(isinstance(transition, Transition) for transition in sample)


# def test_zip():
#     capacity = 10
#     memory = ReplayMemory(capacity)
#     for i in range(10):
#         memory.push(i, i+1, i+2, i+3, False)
#     batch = memory.sample(5)
#     state, action, reward, next_state, done = zip(*batch)
#     assert type(state) is tuple
#     assert len(state) == 5
#     assert all(isinstance(s, int) for s in state)
#     assert all(isinstance(a, int) for a in action)
#     assert all(isinstance(r, int) for r in reward)
#     assert all(isinstance(s, int) for s in next_state)
#     assert all(isinstance(d, bool) for d in done)


# if __name__ == "__main__":
#     test_push_and_len()
#     test_sample()
#     test_zip()
#     print("All tests passed.")
