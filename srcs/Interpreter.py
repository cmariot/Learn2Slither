from Environment import Environment
import numpy as np
import pandas


class Interpreter:

    def interpret(self, environment: Environment):

        """
        This method returns the state of the environment as a numpy array.
        """

        # state = environment.get_state()

        dictionary = {

            # Green apple direction
            'green_apple_up': 0,
            'green_apple_down': 0,
            'green_apple_left': 0,
            'green_apple_right': 0,

            # Red apple direction
            'red_apple_up': 0,
            'red_apple_down': 0,
            'red_apple_left': 0,
            'red_apple_right': 0,

            # Danger direction
            'danger_up': 0,
            'danger_down': 0,
            'danger_left': 0,
            'danger_right': 0,

        }

        dataframe = pandas.DataFrame(dictionary, index=['distance'])
        numpy_array = np.array(dataframe.to_numpy()[0])
        return numpy_array

    def get_reward(self, state, action, snake_alive):
        reward = 0
        return reward
