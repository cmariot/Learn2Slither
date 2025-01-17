from Environment import Environment
import numpy as np
from pandas import DataFrame
from Directions import Directions
from CommandLineInterface import CommandLineInterface
from InterfaceController import InterfaceController
from constants import (
    WALL, GREEN_APPLE, RED_APPLE, SNAKE_BODY, SNAKE_HEAD, EMPTY
)


class Interpreter:

    def interpret(
            self,
            environment: Environment,
            controller: InterfaceController,
            cli: CommandLineInterface,
            is_first_state: bool = False
    ) -> np.ndarray:

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
        no_snake_body = self._have_snake_body(state, x, y, dirs.value_list())

        dictionary = {}
        for key, value in dirs.items():

            green_apple = self._apple_distance(state, x, y, GREEN_APPLE, value)
            red_apple = self._apple_distance(state, x, y, RED_APPLE, value)
            danger = self._danger_distance(state, x, y, value, no_snake_body)
            wall = self._wall_distance(state, x, y, value)

            dictionary[f'green_apple_{key}'] = green_apple
            dictionary[f'red_apple_{key}'] = red_apple
            dictionary[f'danger_{key}'] = danger
            dictionary[f'wall_{key}'] = wall

        dictionary = dict(sorted(dictionary.items()))

        for x in range(0, len(state)):
            for y in range(0, len(state)):
                value = state[x][y]
                dict_snake_int = {
                    EMPTY: ord('0'),
                    SNAKE_HEAD: ord('H'),
                    SNAKE_BODY: ord('S'),
                    GREEN_APPLE: ord('G'),
                    RED_APPLE: ord('R'),
                    WALL: ord('W'),
                    ' ': 0  # Unknown value in the state
                }
                dictionary[f'[{x}][{y}]'] = dict_snake_int[value]

        dataframe = DataFrame(dictionary, columns=dictionary.keys(), index=[0])
        numpy_state = np.array(dataframe, dtype=np.float32)[0]

        cli.save_state(environment, dataframe, controller, is_first_state)

        return numpy_state

    def _apple_distance(self, state, x, y, apple, direction):

        """
        Return the distance between the snake head (x, y) and the apple
        passed as argument (GREEN_APPLE or RED_APPLE) in the direction
        passed as argument.
        """

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

    def _danger_distance(self, state, x, y, direction, no_snake_body):

        """
        Return the minimum distance between the snake head (x, y) and a
        danger (wall, snake body, red apple if no_snake_body is True) in
        the direction passed as argument.
        """

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

    def _wall_distance(self, state, x, y, direction):

        """
        Return the distance between the snake head (x, y) and the wall in the
        direction passed as argument.
        """

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

    def _have_snake_body(self, state, x, y, direction):

        """
        Return True if there no body in ('up', 'down', 'left', 'right')
        direction. Else, return False.
        """

        state_len = len(state)
        if x == 0 or y == 0 or x == state_len - 1 or y == state_len - 1:
            # The snake is at the border of the board
            return True

        return [
            state[y + direction[i][1]][x + direction[i][0]] != SNAKE_BODY
            for i in range(4)
        ]
