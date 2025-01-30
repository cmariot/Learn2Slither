import pyfiglet
from constants import CLEAR, BLUE, RESET, RED
import time
import os


class CommandLineInterface:

    """

    Class CommandLineInterface:

    This class is used to display the game in the terminal. It is used to
    display the game state, the action chosen by the agent, the reward obtained
    and the game over message.

    """

    def __init__(self, args, environment, controller, interpreter):

        """
        Constructor of the CommandLineInterface class.
        """

        self.fps = args.fps
        self.state_str = ""
        self.new_state_str = ""

        if args.no_cli:
            controller.toggle_cli()

        print(CLEAR + BLUE + pyfiglet.figlet_format("Learn2Slither") + RESET)

        print(
            "Welcome to Learn2Slither!\n" +
            "A reinforcement-learning snake game.\n" +
            "The goal of the AI is to be the longest and survive.\n\n" +

            "The environment is a grid of cells where the snake can move.\n" +
            "The snake can only view up, down, left or right.\n" +
            "The state is a 16 value list representing environment cells.\n"
        )

        interpreter.interpret(
            environment, controller, self, True
        )
        self.print_env_state()

        if args.no_cli:
            print("CLI disabled, the game will run in the background")
            print("Press 'c' to enable the CLI")

    def print(
        self, environment, scores, controller,
        reward=None, agent=None
    ):

        if controller.cli_disabled():
            return

        print(CLEAR + BLUE + pyfiglet.figlet_format("Learn2Slither") + RESET)

        if scores is not None:
            print(
                f"Game #{scores.game_number} - " +
                f"Turn #{scores.turn} - " +
                f"Snake length: {scores.snake_len} - " +
                f"Score: {scores.score}\n"
            )
        self.print_env_state()
        self.print_action(agent, reward, controller.is_ai())
        self.print_env_state(is_new_state=True)
        self.print_game_over(environment)

        if controller.gui_disabled():
            time.sleep(1 / self.fps)

    def save_state(
        self, environment, pandas_state, controller, is_first_state
    ):

        """
        Save the state of the environment in a string format to display it
        in the terminal.
        """

        self.__setattr__(
            "state_str" if is_first_state else "new_state_str",
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
            if ('[' in column_name) and (']' in column_name):
                value = f"{column_name}: {chr(int(value.iloc[0]))}"
            else:
                value = f"{column_name}: {value.iloc[0]:.2f}"
            max_value_len = max(max_value_len, len(value))
            if i < len(res):
                lengths.append(len(value))
                res[i] += value
            else:
                idx = i % len(res)
                lengths[idx] = len(value)
                res[idx] += value

            # Add spaces to align the values
            if i % len(res) == len(res) - 1:
                for j in range(len(res)):
                    res[j] += " " * (max_value_len - lengths[j] + 1) + '\t'
                max_value_len = 0

        return "\n".join(res)

    def print_env_state(self, is_new_state=False):
        if is_new_state:
            state = self.new_state_str
        else:
            state = self.state_str
        print(state, end="\n\n")

    def print_action(self, agent, reward, is_ai):

        if agent is None:
            return

        action = ['up', 'down', 'left', 'right'][agent.action]

        if is_ai:
            print(
                f"Agent chose action: {action:<5} " +
                f"(epsilon = {agent.epsilon * 100:.2f}% - " +
                f"{agent.choice_type}) - " +
                f"Reward: {reward}\n"
            )
        else:
            print(
                f"User chose action: {action}. " +
                f"Reward: {reward}\n"
            )

    def print_game_over(self, environment):
        if environment.is_game_over:
            print(f"{BLUE}Game over:{RESET} {environment.game_over_message}\n")
        else:
            print(f"{environment.game_over_message}\n")

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
