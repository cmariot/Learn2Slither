from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from game_mode import GameMode
from Interpreter import Interpreter
from Agent import Agent
import matplotlib.pyplot as plt


def main():

    environment = Environment()
    gui = GraphicalUserInteface(environment)
    game_mode = GameMode("human")
    interpreter = Interpreter()
    agent = Agent()

    print("Press 'space' to switch between human and auto mode")
    print("Press 'q' or 'escape' to exit\n")
    print(f"Gaming in {game_mode} mode\n")

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Score evolution")
    plt.show()

    training = True
    while training:

        environment.reset()

        while environment.is_running():

            gui.handle_key_pressed(environment, game_mode)

            if game_mode.is_ai():
                print(environment)
                state = interpreter.interpret(environment)
                action = agent.choose_action(state)
                is_alive = environment.move_snake(action)
                reward = agent.get_reward(state, action, is_alive)
                environment.score += reward
                print(f"Action: {action}, Reward: {reward}")
                agent.learn(state, action, reward, environment.get_state())

            gui.draw(environment)

            # Clear the previous plot
            ax.clear()
            plt.plot(environment.scores, color="blue")
            fig.canvas.draw()
            plt.pause(0.1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
        plt.ioff()  # TODO: Turn off interactive mode before exiting
