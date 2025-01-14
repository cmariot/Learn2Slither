import pyfiglet
from constants import BLUE, RESET
import time


class CommandLineInterface:

    def __init__(self, fps):
        self.welcome_message()
        self.state_str = ""
        self.new_state_str = ""
        self.fps = fps

    def print(
                self,
                environment,
                score_evolution,
                controller,
                interpreter,
                action=None,
                reward=None
            ):

        if controller.cli_disabled():
            return
        self.welcome_message(score_evolution, environment)
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
                self,
                environment,
                pandas_state,
                controller,
                is_new_state=False
            ):

        if is_new_state:
            self.new_state_str = self.get_state_str(
                environment, pandas_state, controller
            )
        else:
            self.state_str = self.get_state_str(
                environment, pandas_state, controller
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
        max_line_len = 0
        for i, (column_name, value) in enumerate(pandas_state.items()):
            if i < len(res):
                if len(res[i]) > max_line_len:
                    max_line_len = len(res[i])
                res[i] += f"{column_name}: {value.iloc[0]}"
            else:
                idx = i % len(res)
                if len(res[idx]) < max_line_len:
                    spacer = "_" * (len(res[idx]) - max_line_len)
                    res[idx] += spacer
                res[idx] += "\t" + f"{column_name}: {value.iloc[0]}"

        return "\n".join(res)

    def welcome_message(self, score_evolution=None, environment=None):

        # Clear the terminal
        print("\033[H\033[J")

        print(
            f"{BLUE}" +
            pyfiglet.figlet_format("Learn2Slither") +
            f"{RESET}"
        )

        if score_evolution is not None and environment is not None:
            print(
                f"Game #{score_evolution.game_number} - " +
                f"Turn #{score_evolution.turn} - " +
                f"Snake length: {len(environment.snake.body)} - " +
                f"Score: {score_evolution.score}\n"
            )

        print(
            "Welcome to Learn2Slither!\n" +
            "Press 'space' to switch between human and AI mode.\n" +
            "In AI mode, the snake will play by itself.\n" +
            "In human mode, you can control the snake with the arrow keys.\n" +
            "Press 'q', 'esc' or use the close button to exit the game.\n"
        )

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