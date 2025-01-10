from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from game_mode import GameMode
from Interpreter import Interpreter
from Agent import Agent
import pyfiglet
import matplotlib.pyplot as plt
from constants import BLUE, RESET
from Snake import POSITIVE_REWARD
import numpy as np


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

    plt.ion()
    figure, ax = plt.subplots()

    scores = []
    iterations = []
    mean_scores = []

    score_plot, = ax.plot(scores, iterations)
    mean_score_plot, = ax.plot(mean_scores, iterations)

    # Legend, titles and labels
    ax.legend(["Score", "Mean Score"])
    ax.set_title("Learn2Slither score evolution")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Score")

    while training:

        environment.reset()
        agent.current_max_score = 0
        while environment.is_running():

            gui.handle_key_pressed(environment, game_mode)

            if game_mode.is_ai():

                print("/" * 50)
                print(f"Game number: {environment.game_number}\n")

                print(environment)

                # Get the state from the environment
                state = interpreter.interpret(environment)

                # Choose an action based on the state
                action = agent.choose_action(state)
                print(f"\nAction: {action}\n")

                # Perform move and get the new state
                reward, is_alive = environment.move_snake(action)

                if reward == POSITIVE_REWARD:
                    pass
                if reward == -10:
                    print(f"\nReward: {reward}")
                else:
                    print(f"Reward: {reward}")

                new_state = interpreter.interpret(environment)

                # Avoid infinite loops
                print(f"{BLUE}Agent score: ", agent.score, RESET)

                # End the game if the snake is too bad (infinite loops)
                if agent.score < agent.current_max_score * 1.1 - 100:
                    environment.snake.die("Snake is too bad")
                    environment.game_over()
                    is_alive = False
                    reward = agent.score

                # # End the game if the snake is too good (save good snakes)

                elif agent.score > agent.current_max_score * 1.1 + 100:
                    environment.snake.die("Snake is too good")
                    environment.game_over()
                    is_alive = False
                    reward = agent.score

                # Train short memory
                agent.train_short_memory(state, action, reward, new_state, is_alive)

                # Remember
                agent.learn(state, action, reward, new_state, is_alive)

                # Update the best snake score reached
                if agent.score > agent.current_max_score:
                    agent.current_max_score = agent.score

                # Update the high score
                if agent.score > agent.high_score:
                    agent.high_score = agent.score

                if not is_alive:

                    # Train long memory
                    agent.train_long_memory()

                    # Update the score
                    print(f"Score: {agent.score}")
                    # scores.append(agent.score)

                    scores.append(agent.score)
                    mean_scores.append(np.mean(scores))
                    iterations.append(environment.game_number)

                    agent.score = 0
                    score_plot.set_xdata(iterations)
                    score_plot.set_ydata(scores)

                    mean_score_plot.set_xdata(iterations)
                    mean_score_plot.set_ydata(mean_scores)

                    ax.relim()
                    ax.autoscale_view()

                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    plt.pause(0.1)

                print("/" * 50, "\n\n")

            gui.draw(environment, agent.score, agent.high_score)

    plt.ioff()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
