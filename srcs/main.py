from Environment import Environment
from GraphicalUserInterface import GraphicalUserInterface
from CommandLineInterface import CommandLineInterface
from InterfaceController import InterfaceController
from Interpreter import Interpreter
from Agent import Agent
from Score import Score
from ArgumentParser import ArgumentParser
from constants import TRAINING_LOOP, GAMING_LOOP


def main(args: tuple) -> None:

    environment = Environment()
    interpreter = Interpreter()
    agent = Agent(args)

    controller = InterfaceController(args)
    gui = GraphicalUserInterface(environment.height, environment.width, args)
    cli = CommandLineInterface(args)
    score = Score(args)

    cli.print(environment, score, controller, interpreter)

    while TRAINING_LOOP and environment.is_training(gui, score):

        while GAMING_LOOP:

            gui.draw(environment, score, controller)

            # Handle the key pressed by the user
            should_perform_move, action = gui.handle_key_pressed(
                environment, controller, cli, score, agent
            )
            if environment.is_closed:
                break
            elif not should_perform_move:
                continue

            # If the game mode is AI or if the user pressed a key in human mode
            # perform the move

            # Get the current state
            # TODO: combine the two lines below
            state, pandas_state = interpreter.interpret(environment)
            cli.save_state(environment, pandas_state, controller)

            if controller.is_ai():
                # The agent choose an action based on the state
                action = agent.choose_action(state, score.game_number)

            # Perform move and get the reward
            reward, is_alive = environment.move_snake(action)

            # Update the score based on the reward at each step
            score.turn_update(reward, environment.snake.len())

            # Get the new state
            # TODO: combine the two lines below
            new_state, new_pandas_state = interpreter.interpret(environment)
            cli.save_state(environment, new_pandas_state, controller, True)

            # Train the agent based on the new state and the reward
            agent.train(state, action, reward, new_state, is_alive)

            cli.print(
                environment, score, controller, interpreter, action, reward
            )

            if environment.is_game_over:
                gui.game_over(environment, controller)
                agent.train_long_memory()
                score.game_over_update()
                break

        # Reset the environment (snake, food, etc.) at the end of each game
        environment.reset()

    # Save the agent and the scores
    agent.save(score)


if __name__ == "__main__":
    try:
        parser = ArgumentParser()
        args = parser.parse_args()
        main(args)
    except Exception as exception:
        CommandLineInterface.print_exception(exception)
    except KeyboardInterrupt:
        print("\nExiting...")
