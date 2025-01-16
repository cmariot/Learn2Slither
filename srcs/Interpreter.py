from Environment import Environment
import numpy as np
from pandas import DataFrame
from constants import (
    WALL, GREEN_APPLE, RED_APPLE, SNAKE_BODY,
    # SNAKE_HEAD, EMPTY, DEAD_SNAKE
)


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
    state_len = len(state)
    if (
        x == 0 or y == 0 or x == state_len - 1 or y == state_len - 1
    ):
        return True
    return [
        state[y + direction[i][1]][x + direction[i][0]] != SNAKE_BODY
        for i in range(4)
    ]


class Interpreter:

    def interpret(self, environment: Environment):

        """
        This method returns the state of the environment as a numpy array.
        """

        state = environment.get_state()
        direction = ((0, -1), (0, 1), (-1, 0), (1, 0))
        head_x, head_y = environment.snake.get_head_position()
        no_snake_body = have_snake_body(state, head_x, head_y, direction)

        directions = ['up', 'down', 'left', 'right']
        dictionary = {}

        for i, dir in enumerate(directions):
            dictionary[f'green_apple_{dir}'] = apple_distance(
                state, head_x, head_y, GREEN_APPLE, direction[i]
            )
            dictionary[f'red_apple_{dir}'] = apple_distance(
                state, head_x, head_y, RED_APPLE, direction[i]
            )
            dictionary[f'danger_{dir}'] = danger_distance(
                state, head_x, head_y, direction[i], no_snake_body
            )
            dictionary[f'wall_{dir}'] = wall_distance(
                state, head_x, head_y, direction[i]
            )

        # list_x = []
        # list_y = []
        # for x in range(len(state)):
        #     for y in range(len(state)):
        #         if x == head_x and y == head_y:
        #             list_x.append(state[y][x])
        #             list_y.append(state[y][x])
        #         elif x == head_x:
        #             list_x.append(state[y][x])
        #         elif y == head_y:
        #             list_y.append(state[y][x])

        # dictionary['head_x'] = head_x
        # dictionary['head_y'] = head_y
        # int_values = (
        #     GREEN_APPLE,
        #     EMPTY,
        #     RED_APPLE,
        #     WALL,
        #     SNAKE_BODY,
        #     SNAKE_HEAD,
        #     DEAD_SNAKE
        # )

        dictionary = dict(sorted(dictionary.items()))

        # for i, x in enumerate(list_x):
        #     dictionary[f'vertical_{i}'] = int_values.index(x)
        # for i, y in enumerate(list_y):
        #     dictionary[f'horizontal_{i}'] = int_values.index(y)

        dataframe = DataFrame(dictionary, columns=dictionary.keys(), index=[0])
        numpy_state = np.array(dataframe.to_numpy()[0])
        return numpy_state, dataframe
