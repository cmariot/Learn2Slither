from Environment import Environment
import numpy as np
from pandas import DataFrame
from CommandLineInterface import CommandLineInterface
from InterfaceController import InterfaceController
from constants import (
    WALL, GREEN_APPLE, RED_APPLE, SNAKE_BODY,
    SNAKE_HEAD, EMPTY
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

        There are 2 main options in RL for the state representation:
        - Return the entire state of the environment
        - Return the observations as a numpy array of computed features
        In our case, we don't have access to the entire state of the
        environment due to the snake's vision.

        """

        # dirs = Directions()
        state = environment.get_state()
        x_head, y_head = environment.snake.get_head_position()

        # Add all the values of the state to the dictionary
        dictionary = {}
        x_list = []
        y_list = []
        dict_snake_int = {
            EMPTY: ord('0'),
            SNAKE_HEAD: ord('H'),
            SNAKE_BODY: ord('S'),
            GREEN_APPLE: ord('G'),
            RED_APPLE: ord('R'),
            WALL: ord('W'),
            ' ': 0  # Unknown value in the state
        }
        for x in range(0, len(state)):
            for y in range(0, len(state)):
                value = state[x][y]
                if x == x_head:
                    x_list.append(value)
                if y == y_head:
                    y_list.append(value)
                dictionary[f'[{x}][{y}]'] = dict_snake_int[value]

        for x_value in x_list:
            dict

        dataframe = DataFrame(dictionary, columns=dictionary.keys(), index=[0])
        numpy_state = np.array(dataframe, dtype=np.float32)[0]

        cli.save_state(environment, dataframe, controller, is_first_state)

        return numpy_state
