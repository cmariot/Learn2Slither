from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from CommandLineInterface import CommandLineInterface
from InterfaceController import InterfaceController
from Interpreter import Interpreter
from Agent import Agent
from ScoreEvolutionPlot import ScoreEvolutionPlot


TRAINING_LOOP = True
GAMING_LOOP = True


def main():

    # Reinforcement learning variables
    environment = Environment()
    interpreter = Interpreter()
    agent = Agent()

    # Interface variables
    controller = InterfaceController()
    gui = GraphicalUserInteface(environment.height, environment.width)
    cli = CommandLineInterface()
    score_evolution = ScoreEvolutionPlot()

    # TODO:
    # - [ ] Key to disable/enable the GUI
    # - [ ] Command line arguments to load a model
    # - [ ] Command line arguments to set the game mode, training options, etc.

    # - [ ] Training parameters (epsilon, gamma, etc.)
    # - [ ] Exploration vs exploitation
    # - [ ] Split the main function into smaller functions
    # - [ ] Add a logger

    # - [X] Wall distance in the state
    # - [X] Save the model / agent
    # - [X] Fix score/game_number mismatch between the CLI and the GUI
    # - [X] Key to disable/enable the CLI
    # - [X] CLI game over message
    # - [X] GUI game over view with a message

    cli.print(environment, score_evolution, controller, interpreter)

    while TRAINING_LOOP and not gui.is_closed:

        # Reset the environment (snake, food, score, etc.) at the beginning of
        # each game
        environment.reset()

        gui.draw(environment, score_evolution, controller)

        while GAMING_LOOP and not environment.is_game_over:

            # Handle the key pressed by the user
            should_perform_move, action = gui.handle_key_pressed(
                environment, controller, gui, score_evolution    
            )
            if environment.is_closed:
                break
            elif not should_perform_move:
                continue

            # If the game mode is AI or if the user pressed a key in human mode
            # perform the move

            # Get the current state
            state, pandas_state = interpreter.interpret(environment)

            cli.save_state(environment, pandas_state, controller)

            if controller.is_ai():

                # The agent choose an action based on the state
                action = agent.choose_action(
                    state, score_evolution.game_number
                )

            # Perform move and get the reward
            reward, is_alive = environment.move_snake(action)

            # Get the new state
            new_state, new_pandas_state = interpreter.interpret(environment)

            cli.save_state(environment, new_pandas_state, controller, True)

            # Train short memory
            agent.train_short_memory(
                state, action, reward, new_state, is_alive
            )

            # Remember
            agent.learn(
                state, action, reward, new_state, is_alive, score_evolution
            )

            cli.print(
                environment,
                score_evolution,
                controller,
                interpreter,
                action,
                reward
            )

            if environment.is_game_over:

                gui.game_over(environment, controller)

                # Train long memory
                agent.train_long_memory()

                # Update the score plot
                score_evolution.update()

            gui.draw(environment, score_evolution, controller)

    # Save the agent and the score evolution
    agent.save(agent, score_evolution)


if __name__ == "__main__":
    try:
        main()
    # except Exception as e:
        # print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
