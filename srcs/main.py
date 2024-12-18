from Environment import Environment
from GraphicalUserInterface import GraphicalUserInteface
from game_mode import GameMode


def main():

    environment = Environment()
    gui = GraphicalUserInteface(environment)
    game_mode = GameMode("human")

    print("Press 'space' to switch between human and auto mode")
    print("Press 'q' or 'escape' to exit\n")
    print(f"Gaming in {game_mode} mode\n")

    training = True
    while training:

        # TODO: environment.reset()
        environment = Environment()

        game = True
        while game:

            gui.handle_key_pressed(environment, game_mode)

            if game_mode.is_ai():
                # action = board.get_move(state)
                # board.move_snake(action)
                # print(board, "\n")
                # state = board.get_state()
                pass

            gui.draw(environment)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")

# if __name__ == "__main__":
#     main()
