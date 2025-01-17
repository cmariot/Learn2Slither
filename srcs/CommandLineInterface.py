import pyfiglet
from constants import CLEAR, BLUE, RESET, RED
import time
from Environment import Environment
from Interpreter import Interpreter
from InterfaceController import InterfaceController
from Score import Score
import os


class CommandLineInterface:

    """

    Class CommandLineInterface:

    This class is used to display the game in the terminal. It is used to
    display the game state, the action chosen by the agent, the reward obtained
    and the game over message.

    """

    def __init__(self, args):

        """
        Constructor of the CommandLineInterface class.
        """

        self.fps = args.fps
        self.state_str = ""
        self.new_state_str = ""
        self.welcome_message()

    def welcome_message(self, scores: Score = None):

        print(CLEAR + BLUE + pyfiglet.figlet_format("Learn2Slither") + RESET)

        if scores is not None:
            print(
                f"Game #{scores.game_number} - " +
                f"Turn #{scores.turn} - " +
                f"Snake length: {scores.snake_len} - " +
                f"Score: {scores.score}\n"
            )

        print(
            "Welcome to Learn2Slither!\n" +
            "Press 'space' to switch between human and AI mode.\n" +
            "In AI mode, the snake will play by itself.\n" +
            "In human mode, you can control the snake with the arrow keys.\n" +
            "Press 'q', 'esc' or use the close button to exit the game.\n"
        )

    def print(
        self, environment: Environment, scores: Score,
        controller: InterfaceController, interpreter: Interpreter,
        action=None, reward=None
    ):

        if controller.cli_disabled():
            return
        self.welcome_message(scores)
        if action is None and reward is None:
            _, pandas_state = interpreter.interpret(environment)
            self.save_state(environment, pandas_state, controller)
        self.print_env_state()
        if action is not None and reward is not None:
            self.print_action(action, controller.is_ai(), reward)
        self.print_env_state(is_new_state=True)
        if environment.is_game_over:
            self.game_over(environment.game_over_message)
        if controller.gui_disabled():
            time.sleep(1 / self.fps)

    def save_state(
        self, environment, pandas_state, controller, is_new_state=False
    ):

        self.__setattr__(
            "new_state_str" if is_new_state else "state_str",
            self.get_state_str(environment, pandas_state, controller)
        )

    def get_state_str(self, environment, pandas_state, controller):

        if controller.cli_disabled():
            return ""

        res = []
        state = environment.get_state()

        for x in range(environment.width):

            # Print the board
            env_line = ""
            for y in range(environment.height):
                env_line += environment.board[y][x] + " "
            res.append(env_line + '\t')

            # Print the state
            state_line = ""
            for y in range(environment.height):
                state_line += state[x][y] + " "
            res[-1] += state_line + '\t'

        # Append the pandas dataframe at the end of each line
        max_value_len = 0
        lengths = []
        for i, (column_name, value) in enumerate(pandas_state.items()):
            column_name: str = column_name
            if (
                column_name.startswith("horizontal") or
                column_name.startswith("vertical")
            ):
                int_values = ('G', '0', 'R', 'W', 'S', 'H', 'X')
                value = f"{column_name}: {int_values[value.iloc[0]]}"
            else:
                value = f"{column_name}: {value.iloc[0]}"

            if i < len(res):
                lengths.append(len(value))
                if len(value) > max_value_len:
                    max_value_len = len(value)
                res[i] += value
            else:
                idx = i % len(res)
                lengths[idx] = len(value)
                res[idx] += value

            # Add spaces to align the values
            if i % len(res) == len(res) - 1:
                for j in range(len(res)):
                    res[j] += " " * (max_value_len - lengths[j] + 1) + '\t'

        return "\n".join(res)

    def print_env_state(self, is_new_state=False):
        if is_new_state:
            state = self.new_state_str
        else:
            state = self.state_str
        print(state, end="\n\n")

    def print_action(self, action, is_ai, reward):
        action = ['up', 'down', 'left', 'right'][action]
        if is_ai:
            print(f"Agent chose action: {action}")
        else:
            print(f"User chose action: {action}")
        print(f"Reward: {reward}\n")

    def game_over(self, game_over_message):
        print(f"{BLUE}Game over:{RESET} {game_over_message}\n")

    def set_fps(self, fps):
        self.fps = fps

    def print_exception(exception: Exception) -> None:

        """
        Prints detailed information about an exception, including its type,
        message, and a detailed traceback with file, line number, and code
        context.
        """

        if not isinstance(exception, Exception):
            raise TypeError("The argument should be an exception.")

        print(f"{RED}An exception occurred:{RESET}\n")

        # Print exception type and message
        print(f"{RED}Exception Type:{RESET} {type(exception).__name__}")
        print(f"{RED}Exception Message:{RESET} {exception}\n")

        print(f"{RED}Traceback (most recent call last):{RESET}\n")

        i = 0
        tb = exception.__traceback__
        current_directory = os.getcwd()

        while tb:
            filename = tb.tb_frame.f_code.co_filename
            relative_filename = filename.replace(current_directory, ".")
            line = tb.tb_lineno
            function_name = tb.tb_frame.f_code.co_name
            if function_name == "<module>":
                function_name = "Main Context"

            # Retrieve and print the relevant line of code
            try:
                with open(filename, "r") as file:
                    lines = file.readlines()
                    code_context = lines[line - 1].strip()
            except (FileNotFoundError, IndexError):
                code_context = "Unable to retrieve code context."

            # Format the traceback information
            print(
                f"{i}:  File: {RED}\"{relative_filename}\"{RESET}, "
                f"Line: {RED}{line}{RESET}, " +
                f"Function: {RED}{function_name}{RESET}\n" +
                f"      Code Context: {RED}{code_context}{RESET}\n"
            )

            # Move to the next traceback level
            tb = tb.tb_next
            i += 1

        print(f"{RED}End of Traceback.{RESET}")
