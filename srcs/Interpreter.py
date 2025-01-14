from Environment import Environment
import numpy as np
import pandas
from constants import WALL, GREEN_APPLE, RED_APPLE, SNAKE_BODY


class Interpreter:

    def interpret(self, environment: Environment):

        """
        This method returns the state of the environment as a numpy array.
        """

        state = environment.get_state()

        head_x, head_y = environment.snake.get_head_position()

        def apple_distance(state, x, y, apple, direction):
            distance = 0
            while state[y][x] != apple and state[y][x] != WALL:
                distance += 1
                x += direction[0]
                y += direction[1]
            if state[y][x] == WALL:
                return 0
            return distance

        def danger_distance(state, x, y, direction, no_snake_body):
            distance = 0
            while (
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
            while state[y][x] != WALL:
                distance += 1
                x += direction[0]
                y += direction[1]
            return distance

        # Check if there is a snake body part at position (x, y) + direction
        direction = ((0, -1), (0, 1), (-1, 0), (1, 0))
        no_snake_body = [
            state[head_y + direction[i][1]][head_x + direction[i][0]] !=
            SNAKE_BODY
            for i in range(4)
        ]
        no_snake_body = True if sum(no_snake_body) == 4 else False

        dictionary = {

            # Green apple direction
            'green_apple_up': apple_distance(
                state, head_x, head_y, GREEN_APPLE, direction[0]
            ),
            'green_apple_down': apple_distance(
                state, head_x, head_y, GREEN_APPLE, direction[1]
            ),
            'green_apple_left': apple_distance(
                state, head_x, head_y, GREEN_APPLE, direction[2]
            ),
            'green_apple_right': apple_distance(
                state, head_x, head_y, GREEN_APPLE, direction[3]
            ),

            # Red apple direction
            'red_apple_up': apple_distance(
                state, head_x, head_y, RED_APPLE, direction[0]
            ),
            'red_apple_down': apple_distance(
                state, head_x, head_y, RED_APPLE, direction[1]
            ),
            'red_apple_left': apple_distance(
                state, head_x, head_y, RED_APPLE, direction[2]
            ),
            'red_apple_right': apple_distance(
                state, head_x, head_y, RED_APPLE, direction[3]
            ),

            # Danger direction
            'danger_up': danger_distance(
                state, head_x, head_y, direction[0], no_snake_body
            ),
            'danger_down': danger_distance(
                state, head_x, head_y, direction[1], no_snake_body
            ),
            'danger_left': danger_distance(
                state, head_x, head_y, direction[2], no_snake_body
            ),
            'danger_right': danger_distance(
                state, head_x, head_y, direction[3], no_snake_body
            ),

            # Wall direction
            'wall_up': wall_distance(state, head_x, head_y, direction[0]),
            'wall_down': wall_distance(state, head_x, head_y, direction[1]),
            'wall_left': wall_distance(state, head_x, head_y, direction[2]),
            'wall_right': wall_distance(state, head_x, head_y, direction[3]),

        }

        dataframe = pandas.DataFrame(
            dictionary, columns=dictionary.keys(), index=[0]
        )
        numpy_state = np.array(dataframe.to_numpy()[0])
        pandas_state = dataframe
        return numpy_state, pandas_state
