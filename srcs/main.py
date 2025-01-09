from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from game_mode import GameMode
from Interpreter import Interpreter
from Agent import Agent
import pyfiglet


def header(game_mode):
    print(pyfiglet.figlet_format("Learn2Slither"))
    print("Welcome to Learn2Slither!")
    print("Move the snake with the arrow keys")
    print("Press 'space' to switch between human and auto mode")
    print("Press 'q' or 'escape' to exit\n")
    print(f"Gaming in {game_mode} mode\n")


def main():

    environment = Environment()
    interpreter = Interpreter()
    agent = Agent()
    game_mode = GameMode("human")
    gui = GraphicalUserInteface(environment)

    header(game_mode)

    training = True
    while training:

        environment.reset()

        while environment.is_running():

            gui.handle_key_pressed(environment, game_mode)

            if game_mode.is_ai():
                state = interpreter.interpret(environment)
                action = agent.choose_action(state, environment.game_number)
                is_alive = environment.move_snake(action)
                reward = interpreter.get_reward(state, action, is_alive)

                agent.learn(
                    state,
                    action,
                    reward,
                    interpreter.interpret(environment),
                    is_alive
                )

            gui.draw(environment)
            environment.score += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
