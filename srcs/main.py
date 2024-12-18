from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from game_mode import GameMode
from Interpreter import Interpreter
from Agent import Agent


def main():

    environment = Environment()
    gui = GraphicalUserInteface(environment)
    game_mode = GameMode("human")
    interpreter = Interpreter()
    agent = Agent()

    print("Press 'space' to switch between human and auto mode")
    print("Press 'q' or 'escape' to exit\n")
    print(f"Gaming in {game_mode} mode\n")

    training = True
    while training:

        # TODO: environment.reset()
        environment = Environment()

        while environment.is_running():

            gui.handle_key_pressed(environment, game_mode)

            if game_mode.is_ai():
                print(environment)
                state = interpreter.interpret(environment)
                action = agent.choose_action(state)
                is_alive = environment.move_snake(action)
                reward = agent.get_reward(state, action, is_alive)
                print(f"Action: {action}, Reward: {reward}")
                # agent.learn(state, action, reward, environment.get_state())

            gui.draw(environment)


if __name__ == "__main__":
    try:
        main()
    # except Exception as e:
    #     print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
