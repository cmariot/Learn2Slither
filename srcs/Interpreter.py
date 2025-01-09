from Environment import Environment
import numpy as np
import pandas
from constants import EMPTY, WALL, GREEN_APPLE, RED_APPLE, SNAKE_BODY, SNAKE_HEAD


class Interpreter:

    def interpret(self, environment: Environment):

        """
        This method returns the state of the environment as a numpy array.
        """

        state = environment.get_state()

        # for row in state:
        #     for cell in row:
        #         print(cell, end=' ')
        #     print()

        head_x, head_y = environment.snake.get_head_position()

        def danger_detected(state, x, y, no_snake_body):
            collision = int(
                state[y][x] == WALL or
                state[y][x] == SNAKE_BODY or
                (no_snake_body and state[y][x] == RED_APPLE)
            )
            return collision

        def apple_distance(state, x, y, apple, direction):

            distance = 0

            while state[y][x] != apple and state[y][x] != WALL:
                distance += 1
                x += direction[0]
                y += direction[1]

            if state[y][x] == WALL:
                return 0

            return distance

        GREEN = GREEN_APPLE
        RED = RED_APPLE

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
                state, head_x, head_y, GREEN, direction[0]
            ),
            'green_apple_down': apple_distance(
                state, head_x, head_y, GREEN, direction[1]
            ),
            'green_apple_left': apple_distance(
                state, head_x, head_y, GREEN, direction[2]
            ),
            'green_apple_right': apple_distance(
                state, head_x, head_y, GREEN, direction[3]
            ),

            # Red apple direction
            'red_apple_up': apple_distance(
                state, head_x, head_y, RED, direction[0]
            ),
            'red_apple_down': apple_distance(
                state, head_x, head_y, RED, direction[1]
            ),
            'red_apple_left': apple_distance(
                state, head_x, head_y, RED, direction[2]
            ),
            'red_apple_right': apple_distance(
                state, head_x, head_y, RED, direction[3]
            ),

            # Danger direction
            'danger_up': danger_detected(
                state, head_x, head_y - 1, no_snake_body
            ),
            'danger_down': danger_detected(
                state, head_x, head_y + 1, no_snake_body
            ),
            'danger_left': danger_detected(
                state, head_x - 1, head_y, no_snake_body
            ),
            'danger_right': danger_detected(
                state, head_x + 1, head_y, no_snake_body
            ),

        }

        dataframe = pandas.DataFrame(dictionary, index=['distance'])
        print(dataframe.T)
        numpy_array = np.array(dataframe.to_numpy()[0])
        return numpy_array

    def get_reward(self, state, action, snake_alive):
        reward = 0
        return reward
