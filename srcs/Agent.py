import random


class Agent:

    def __init__(self):
        pass

    def choose_action(self, state):
        actions = ("up", "down", "left", "right")
        return random.choice(actions)

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
            reward -= 5
        elif green_apple_distance == 1:
            reward += 5
        elif empty_distance == 1:
            reward -= 1
        elif (
            red_apple_distance == 1 and not snake_alive or
            wall_distance == 1 or
            body_distance == 1
        ):
            reward -= 42

        return reward

    def learn(self, state, action, reward, next_state):
        pass
