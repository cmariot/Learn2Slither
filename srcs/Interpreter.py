from Environment import Environment
from constants import SNAKE_HEAD, SNAKE_BODY, WALL, GREEN_APPLE, RED_APPLE, EMPTY
import pandas
import numpy as np


class Interpreter:

    def interpret(self, environment: Environment):
        state = environment.get_state()

        for x in range(len(state)):
            for y in range(len(state[x])):
                if state[y][x] == SNAKE_HEAD:
                    head = (x, y)
                    break

        max_distance = len(state)

        dictionary = {

            # Green apple distance
            'up_green_apple': max_distance,
            'down_green_apple': max_distance,
            'left_green_apple': max_distance,
            'right_green_apple': max_distance,

            # Red apple distance
            'up_red_apple': max_distance,
            'down_red_apple': max_distance,
            'left_red_apple': max_distance,
            'right_red_apple': max_distance,

            # Wall distance
            'up_wall': max_distance,
            'down_wall': max_distance,
            'left_wall': max_distance,
            'right_wall': max_distance,

            # Body distance
            'up_body': max_distance,
            'down_body': max_distance,
            'left_body': max_distance,
            'right_body': max_distance,

            # Empty distance
            'up_empty': max_distance,
            'down_empty': max_distance,
            'left_empty': max_distance,
            'right_empty': max_distance

        }

        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
        }

        for direction, (dx, dy) in directions.items():
            x, y = head
            distance = 0
            while (
                x >= 0 and x < len(state[0]) - 1 and
                y >= 0 and y < len(state) - 1
            ):
                x += dx
                y += dy
                distance = 1
                if (
                    state[y][x] == WALL and
                    distance < dictionary[f'{direction}_wall']
                ):
                    dictionary[f'{direction}_wall'] = distance
                elif (
                    state[y][x] == SNAKE_BODY and
                    distance < dictionary[f'{direction}_body']
                ):
                    dictionary[f'{direction}_body'] = distance
                elif (
                    state[y][x] == GREEN_APPLE and
                    distance < dictionary[f'{direction}_green_apple']
                ):
                    dictionary[f'{direction}_green_apple'] = distance
                elif (
                    state[y][x] == RED_APPLE and
                    distance < dictionary[f'{direction}_red_apple']
                ):
                    dictionary[f'{direction}_red_apple'] = distance
                elif (
                    state[y][x] == EMPTY and
                    distance < dictionary[f'{direction}_empty']
                ):
                    dictionary[f'{direction}_empty'] = distance

        # Replace max_distance with 0 for better readability
        for key, value in dictionary.items():
            if value == max_distance:
                dictionary[key] = 0

        # Print the dataframe
        dataframe = pandas.DataFrame(dictionary, index=['distance'])
        print(dataframe.transpose(), '\n')

        # print(type(dataframe.to_numpy()))
        # print(dataframe.to_numpy()[0])
        # exit()
        numpy_array = np.array(dataframe.to_numpy()[0])

        return numpy_array
