from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from CommandLineInterface import CommandLineInterface
from InterfaceController import InterfaceController
from Interpreter import Interpreter
from Agent import Agent
from ScoreEvolutionPlot import ScoreEvolutionPlot
from ArgumentParser import ArgumentParser


TRAINING_LOOP = True
GAMING_LOOP = True


def main():

    # Parse the arguments
    args = ArgumentParser().args

    # training_sessions = args.training_sessions
    # fps = args.fps
    # step_by_step = args.step_by_step
    # model_path = args.model_path
    # dont_save = args.dont_save
    # new_model = args.new_model
    # is_training = args.train
    # display_plot = args.plot

    # Reinforcement learning variables
    environment = Environment()
    interpreter = Interpreter()
    agent = Agent(args)

    # Interface variables
    controller = InterfaceController(args)
    gui = GraphicalUserInteface(environment.height, environment.width, args)
    cli = CommandLineInterface(args)
    score_evolution = ScoreEvolutionPlot(args)

    # TODO:
    # - [ ] Train the model with the base state ?
    # - [ ] A* algorithm to determine the min snake length ?
    # - [X] Key to enable/disable the step by step mode
    # - [X] Load a non-existant model error
    # - [X] Dont train argument
    # - [X] Game number decrease when the model is saved
    # - [X] Key to save the model / scores

    cli.print(environment, score_evolution, controller, interpreter)

    while (
        TRAINING_LOOP and
        not gui.is_closed and
        score_evolution.training_session_not_finished()
    ):

        # Reset the environment (snake, food, score, etc.) at the beginning of
        # each game
        environment.reset()

        gui.draw(environment, score_evolution, controller)

        while GAMING_LOOP and not environment.is_game_over:

            # Handle the key pressed by the user

            # Key mapping:
            # - Use 'space' to switch between AI and Human mode
            # - Use 'c' to enable/disable the CLI
            # - Use 'g' to enable/disable the GUI
            # - Use 's' to enable/disable the step by step mode
            #   (only in AI mode)
            # - Use 'q', 'esc' or use the close button to quit the game
            # - '+' and '-' to increase or decrease the FPS (+/- 10 fps)
            # - In AI Mode, if the step by step mode is enabled, press 'enter'
            #   to perform the next move

            should_perform_move, action = gui.handle_key_pressed(
                environment, controller, gui, cli, score_evolution, agent
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

            # Update the score evolution
            score_evolution.turn_update(reward, len(environment.snake.body))

            # Get the new state
            new_state, new_pandas_state = interpreter.interpret(environment)

            cli.save_state(environment, new_pandas_state, controller, True)

            # Train short memory
            agent.train_short_memory(
                state, action, reward, new_state, is_alive
            )

            # Remember
            agent.learn(state, action, reward, new_state, is_alive)

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
                score_evolution.game_over_update()

            gui.draw(environment, score_evolution, controller)

    # Save the agent and the score evolution
    agent.save(agent, score_evolution)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
