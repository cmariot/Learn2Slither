from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from game_mode import GameMode
from Interpreter import Interpreter
from Agent import Agent
# import random


def main():

    environment = Environment()
    gui = GraphicalUserInteface(environment)
    game_mode = GameMode("human")
    interpreter = Interpreter()
    agent = Agent()

    print("Press 'space' to switch between human and auto mode")
    print("Press 'q' or 'escape' to exit\n")
    print(f"Gaming in {game_mode} mode\n")

    scores = []
    # mean_scores = []

    training = True
    while training:

        score = 0
        environment.reset()

        while environment.is_running():

            gui.handle_key_pressed(environment, game_mode)

            if game_mode.is_ai():
                print(environment)
                state = interpreter.interpret(environment)
                action = agent.choose_action(state, environment.nb_games)
                is_alive = environment.move_snake(action)
                reward = agent.get_reward(state, action, is_alive)
                score += reward
                new_state = interpreter.interpret(environment)
                action = {'up': 0, 'down': 1, 'left': 2, 'right': 3}[action]
                agent.train_short_memory(
                    state,
                    action,
                    reward,
                    new_state,
                    is_alive
                )
                agent.learn(
                    state,
                    action,
                    reward,
                    new_state,
                    is_alive
                )
                if not is_alive:
                    agent.train_long_memory()

                    scores.append(score)

                    # Save the scores in a file
                    # score = len(environment.snake.body)
                    import numpy as np
                    with open("scores.txt", "a") as f:
                        f.write(f"{int(np.mean(scores))}\n")

                    # break

                    # mean_score = sum(scores) / len(scores)
                    # mean_scores.append(mean_score)

            gui.draw(environment)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
