from Environment import Environment
from GraphicalUserInterface import GraphicalUserInterface
from CommandLineInterface import CommandLineInterface
from InterfaceController import InterfaceController
from Interpreter import Interpreter
from Agent import Agent
from Score import Score
from ArgumentParser import ArgumentParser
from constants import TRAINING_LOOP, GAMING_LOOP

CONTINUE = 0
BREAK = 1


def play_step(
    environment: Environment,
    interpreter: Interpreter,
    agent: Agent,
    controller: InterfaceController,
    gui: GraphicalUserInterface,
    cli: CommandLineInterface,
    score: Score,
    action: int = None
) -> int:

    """
    This function is called at each turn of the game.
    It handles the training of the agent in the gaming loop.
    """

    if gui.is_closed():
        return BREAK

    # Get the current state of the game
    state = interpreter.interpret(environment, controller, cli, True)

    if controller.is_ai():
        # The agent choose an action based on the state
        action = agent.choose_action(state, score.game_number)

    # Perform move and get the reward
    reward = environment.move_snake(action)

    # Update the score based on the reward at each step
    score.turn_update(reward, environment.snake.len())

    # Get the new state
    new_state = interpreter.interpret(environment, controller, cli)

    # Train the agent based on the new state and the reward
    agent.train(state, action, reward, new_state, environment.is_game_over)

    cli.print(environment, score, controller, reward, agent)

    if environment.is_game_over:
        gui.game.game_over(environment, controller, gui)
        agent.train_long_memory()
        score.game_over_update()
        return BREAK

    return CONTINUE


def game(
        environment: Environment,
        controller: InterfaceController,
        gui: GraphicalUserInterface,
        score: Score,
        agent: Agent,
        interpreter: Interpreter,
        cli
) -> None:

    while TRAINING_LOOP and environment.is_training(gui, score):
        while GAMING_LOOP and not gui.is_closed():
            gui.game.draw(environment, score, controller)
            should_perform_move, action = gui.game.handle_key_pressed(
                environment, controller, cli, score, agent, gui)
            if not should_perform_move:
                continue
            if score.should_save_periodically(1000):
                agent.save(score)
            if play_step(
                environment, interpreter, agent, controller,
                gui, cli, score, action
            ) != CONTINUE:
                break
        environment.reset()
    agent.save(score)


def main(args: tuple) -> None:

    """
    Main function of Learn2Slither.
    It initializes the environment, the interpreter, the agent,
    the controller, the GUI, the CLI and the score.
    It handles the training and the gaming loops.
    """

    environment = Environment(args)
    interpreter = Interpreter()
    agent = Agent(args)

    controller = InterfaceController(args)
    gui = GraphicalUserInterface(environment.height, environment.width, args)
    score = Score(args)
    cli = CommandLineInterface(
        args, environment, score, controller, interpreter
    )

    while True:

        if gui.state == "MENU":
            gui.menu.draw_menu()
            if gui.menu.handle_menu_events(gui) == "EXIT":
                break
        elif gui.state == "GAME":
            game(environment, controller, gui, score, agent, interpreter, cli)
            break
        elif gui.state == "SETTINGS":
            # gui.draw_settings()
            # gui.handle_settings_events()
            pass
        elif gui.state == "EXIT":
            break
        else:
            print(f"GUI.State: {gui.state}")
            break


if __name__ == "__main__":
    try:
        parser = ArgumentParser()
        args = parser.parse_args()
        main(args)
    except Exception as exception:
        CommandLineInterface.print_exception(exception)
    except KeyboardInterrupt:
        print("\nExiting...")
