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

    with plt.ion():

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Score evolution")
        x = []
        y = []
        li, = ax.plot(x, y)
        fig.canvas.draw()
        plt.show(block=False)

        training = True
        while training:

            environment.reset()

            while environment.is_running():

                gui.handle_key_pressed(environment, game_mode)

                if game_mode.is_ai():
                    print(environment)
                    state = interpreter.interpret(environment)
                    action = agent.choose_action(state, environment.nb_games)
                    is_alive = environment.move_snake(action)
                    reward = agent.get_reward(state, action, is_alive)
                    environment.score += 1
                    agent.learn(
                        state,
                        action,
                        reward,
                        environment.get_state(),
                        is_alive
                    )

                gui.draw(environment)

            x.append(environment.game_number)
            y.append(environment.score)
            li.set_xdata(x)
            li.set_ydata(y)
            ax.relim()
            ax.autoscale_view(True, True, True)
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
