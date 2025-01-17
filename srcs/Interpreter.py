from Environment import Environment
import numpy as np
from pandas import DataFrame
from constants import (WALL, GREEN_APPLE, RED_APPLE, SNAKE_BODY)
from Directions import Directions


def apple_distance(state, x, y, apple, direction):
    distance = 0
    state_len = len(state)
    while (
        x > 0 and y > 0 and x < state_len - 1 and y < state_len - 1 and
        state[y][x] != apple and
        state[y][x] != WALL
    ):
        distance += 1
        x += direction[0]
        y += direction[1]
    if state[y][x] == WALL:
        return 0
    return distance


def danger_distance(state, x, y, direction, no_snake_body):
    distance = 0
    state_len = len(state)
    while (
        x > 0 and y > 0 and x < state_len - 1 and y < state_len - 1 and
        state[y][x] != WALL and
        state[y][x] != SNAKE_BODY and
        not (state[y][x] == RED_APPLE and no_snake_body)
    ):
        distance += 1
        x += direction[0]
        y += direction[1]
    return distance


def wall_distance(state, x, y, direction):
    distance = 0
    state_len = len(state)
    while (
        x > 0 and y > 0 and x < state_len - 1 and y < state_len - 1 and
        state[y][x] != WALL
    ):
        distance += 1
        x += direction[0]
        y += direction[1]
    return distance


def have_snake_body(state, x, y, direction):

    """
    Return True if there no body in ('up', 'down', 'left', 'right') direction.
    Else, return False.
    """

    state_len = len(state)
    if x == 0 or y == 0 or x == state_len - 1 or y == state_len - 1:
        # The snake is at the border of the board
        return True

    return [
        state[y + direction[i][1]][x + direction[i][0]] != SNAKE_BODY
        for i in range(4)
    ]


class Interpreter:

    def interpret(self, environment: Environment):

        """
        This method returns the state of the environment as a numpy array.
        The state is a 1D array of 16 elements.
        4 elements for each direction (up, down, left, right).
        - Distance to the green apple in that direction.
        - Distance to the red apple in that direction.
        - Distance to the wall in that direction.
        - Distance to the danger in that direction.
        """

        dirs = Directions()
        state = environment.get_state()
        x, y = environment.snake.get_head_position()
        no_snake_body = have_snake_body(state, x, y, dirs.value_list())

        dictionary = {}
        for key, value in dirs.items():
            dictionary[f'green_apple_{key}'] = apple_distance(
                state, x, y, GREEN_APPLE, value
            )
            dictionary[f'red_apple_{key}'] = apple_distance(
                state, x, y, RED_APPLE, value
            )
            dictionary[f'danger_{key}'] = danger_distance(
                state, x, y, value, no_snake_body
            )
            dictionary[f'wall_{key}'] = wall_distance(
                state, x, y, value
            )

        dictionary = dict(sorted(dictionary.items()))
        dataframe = DataFrame(dictionary, columns=dictionary.keys(), index=[0])
        numpy_state = np.array(dataframe, dtype=np.float32)[0]
        return numpy_state, dataframe
