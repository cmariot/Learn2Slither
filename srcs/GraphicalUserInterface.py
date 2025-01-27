from Environment import Environment
from constants import RED_APPLE, GREEN_APPLE, SNAKE_HEAD, SNAKE_BODY, WALL
import pygame
from InterfaceController import InterfaceController
from CommandLineInterface import CommandLineInterface
from Score import Score
from Agent import Agent


class GraphicalUserInterface:

    CELL_SIZE = 40

    def __init__(self, board_width, board_height, args):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.board_width = board_width
        self.board_height = board_height
        self.window_width = board_width * self.CELL_SIZE
        self.window_height = board_height * self.CELL_SIZE
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        self.load_snake_images()
        self.load_apple_images()
        pygame.display.set_caption("Learn2Slither")
        pygame.font.init()
        self.font = pygame.font.get_default_font()
        self._is_closed = False
        self.fps = args.fps

    def load_snake_images(self):

        # Load the asset image
        self.snake_image = pygame.image.load("srcs/assets/snake.png")

        # Define the coordinates for each part of the snake
        parts = {
            'snake_head_up': (120, 0), 'snake_head_right': (160, 0),
            'snake_head_down': (120, 40), 'snake_head_left': (160, 40),
            'body_vertical': (80, 0), 'body_horizontal': (80, 40),
            'body_top_left': (200, 0), 'body_bottom_left': (200, 40),
            'body_bottom_right': (240, 0), 'body_top_right': (240, 40),
            'snake_tail_up': (0, 0), 'snake_tail_right': (40, 0),
            'snake_tail_left': (0, 40), 'snake_tail_down': (40, 40)
        }

        # Use a loop to create subsurfaces for each part
        for part, (x, y) in parts.items():
            setattr(self, part, self.snake_image.subsurface((x, y, 40, 40)))
        self.rescale_images()

    def load_apple_images(self):

        apples = (
            ('green_apple', 'srcs/assets/green_apple.png'),
            ('red_apple', 'srcs/assets/red_apple.png')
        )

        for apple, path in apples:
            setattr(self, apple, pygame.image.load(path))
            # Rescale the images
            setattr(
                self, apple,
                pygame.transform.scale(
                    getattr(self, apple),
                    (self.CELL_SIZE * 0.5, self.CELL_SIZE * 0.5)
                )
            )

    def rescale_images(self):

        # Scale all the images to the cell size

        images = [
            'snake_head_up', 'snake_head_right', 'snake_head_down',
            'snake_head_left', 'body_vertical', 'body_horizontal',
            'body_top_left', 'body_bottom_left', 'body_bottom_right',
            'body_top_right', 'snake_tail_up', 'snake_tail_right',
            'snake_tail_left', 'snake_tail_down'
        ]

        for image in images:
            setattr(
                self, image,
                pygame.transform.scale(
                    getattr(self, image),
                    (self.CELL_SIZE, self.CELL_SIZE)
                )
            )

    def handle_key_pressed(
        self,
        environment: Environment,
        controller: InterfaceController,
        cli: CommandLineInterface,
        scores: Score,
        agent: Agent
    ) -> tuple[bool, int]:

        """
        Handle the key pressed by the user.
        Return a tuple with a boolean to indicate if the move should be
        performed and the action to perform as an integer.
        """

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return self.close()

            elif event.type == pygame.KEYDOWN:

                key = pygame.key.name(event.key)

                if (
                    key in ('return', 'enter') and
                    controller.is_ai() and
                    controller.step_by_step
                ):
                    # Perform the next move in step by step mode
                    return True, None

                elif key == 'space':
                    # Switch between AI and Human mode
                    controller.toggle_ai()

                elif key == 'c':
                    # Enable/disable the CLI
                    controller.toggle_cli()

                elif key == 'g':
                    # Enable/disable the GUI
                    controller.toggle_gui(self, environment, scores)

                elif key == 'p':
                    # Enable/disable the step by step mode
                    controller.toggle_step_by_step()

                elif key == 'q' or key == 'escape':
                    return self.close()

                elif key in ('[+]', '[-]', '-', '='):
                    # Increase or decrease the FPS (+/- 10 fps)
                    shift_pressed = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    controller.change_fps(key, self, cli, shift_pressed)

                elif key == 's':
                    # Save the model and the score evolution
                    agent.save(scores)

                elif controller.is_human():
                    # Handle the key pressed by the user
                    if key in ("up", "down", "left", "right"):
                        key = ('up', 'down', 'left', 'right').index(key)
                        agent.action = key
                        return True, key

        if controller.step_by_step and controller.is_ai():
            return False, None

        return controller.is_ai(), None

    def draw(
        self,
        environment: Environment,
        scores: Score,
        controller: InterfaceController
    ):

        """
        Draw the game board on the pygame window
        """

        if controller.gui_disabled():
            return

        self.screen.fill((0, 0, 0))
        for y, row in enumerate(environment.board):
            for x, cell in enumerate(row):
                cell = environment.board[x][y]
                if (x + y) % 2 == 0:
                    cell_color = (170, 215, 81)
                else:
                    cell_color = (162, 209, 73)
                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    (
                        x * self.CELL_SIZE, y * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                )
                if cell == SNAKE_HEAD or cell == SNAKE_BODY:

                    # Draw the snake image
                    # Get the direction of the snake
                    if cell == SNAKE_HEAD:

                        # Get the direction of the snake based on the body list
                        direction = environment.snake.direction

                        image = {
                            (0, -1): self.snake_head_up,
                            (0, 1): self.snake_head_down,
                            (-1, 0): self.snake_head_left,
                            (1, 0): self.snake_head_right
                        }[direction]

                        self.screen.blit(
                            image,
                            (x * self.CELL_SIZE, y * self.CELL_SIZE)
                        )

                    else:

                        snake_length = environment.snake.len()
                        body_index = environment.snake.get_body_index(x, y)

                        # Tail of the snake
                        if body_index == snake_length - 1:

                            (
                                previous_body_x, previous_body_y
                            ) = environment.snake.body[-2]

                            if previous_body_x == x and previous_body_y < y:
                                image = self.snake_tail_up
                            elif previous_body_x == x and previous_body_y > y:
                                image = self.snake_tail_down
                            elif previous_body_x < x and previous_body_y == y:
                                image = self.snake_tail_left
                            else:
                                image = self.snake_tail_right
                            self.screen.blit(
                                image,
                                (x * self.CELL_SIZE, y * self.CELL_SIZE)
                            )

                        # Body of the snake
                        else:

                            (
                                previous_body_x, previous_body_y
                            ) = environment.snake.body[body_index + 1]
                            (
                                next_body_x, next_body_y
                            ) = environment.snake.body[body_index - 1]

                            # Horizontal body
                            if (
                                previous_body_x < x and
                                next_body_x > x or previous_body_x > x and
                                next_body_x < x
                            ):
                                image = self.body_horizontal

                            # Vertical body
                            elif (
                                previous_body_y < y and
                                next_body_y > y or
                                previous_body_y > y and
                                next_body_y < y
                            ):
                                image = self.body_vertical

                            # Top right corner
                            elif (
                                previous_body_x < x and
                                next_body_y < y or
                                previous_body_y < y and
                                next_body_x < x
                            ):
                                image = self.body_top_right

                            # Top left corner
                            elif (
                                previous_body_x > x and
                                next_body_y < y or
                                previous_body_y < y and
                                next_body_x > x
                            ):
                                image = self.body_top_left

                            # Bottom right corner
                            elif (
                                previous_body_x < x and
                                next_body_y > y or
                                previous_body_y > y and
                                next_body_x < x
                            ):
                                image = self.body_bottom_right

                            # Bottom left corner
                            else:
                                image = self.body_bottom_left

                            self.screen.blit(
                                image,
                                (x * self.CELL_SIZE, y * self.CELL_SIZE)
                            )
                elif cell == RED_APPLE:
                    self.screen.blit(
                        self.red_apple,
                        (
                            x * self.CELL_SIZE + 0.25 * self.CELL_SIZE,
                            y * self.CELL_SIZE + 0.25 * self.CELL_SIZE
                        )
                    )
                elif cell == GREEN_APPLE:
                    self.screen.blit(
                        self.green_apple,
                        (
                            x * self.CELL_SIZE + 0.25 * self.CELL_SIZE,
                            y * self.CELL_SIZE + 0.25 * self.CELL_SIZE
                        )
                    )
                elif cell == WALL:
                    pygame.draw.rect(
                        self.screen,
                        (87, 138, 52),
                        (
                            x * self.CELL_SIZE, y * self.CELL_SIZE,
                            self.CELL_SIZE, self.CELL_SIZE
                        )
                    )

        # Display the score on the screen
        font = pygame.font.Font(self.font, 24)
        score_text = font.render(
            f"Score: {scores.snake_len}",
            True,
            (170, 215, 81)
        )
        self.screen.blit(score_text, (10, 10))

        # Display the high score on the screen
        high_score_text = font.render(
            f"High Score: {scores.high_score}",
            True,
            (170, 215, 81)
        )
        WINDOW_WIDTH = environment.width * self.CELL_SIZE
        x = WINDOW_WIDTH - high_score_text.get_width() - 10
        y = 10
        self.screen.blit(high_score_text, (x, y))

        # Display the game_number on the screen (center bottom)
        game_number_text = font.render(
            f"Game Number: {scores.game_number}",
            True,
            (170, 215, 81)
        )
        x = (WINDOW_WIDTH - game_number_text.get_width()) / 2
        y = WINDOW_WIDTH - game_number_text.get_height() - 10
        self.screen.blit(game_number_text, (x, y))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def game_over(self, environment, controller: InterfaceController):

        if controller.gui_disabled():
            return

        # Display a transparent rectangle on the screen
        transparent = pygame.Surface(
            ((environment.width - 2) * self.CELL_SIZE,
             (environment.height - 2) * self.CELL_SIZE)
        )
        transparent.set_alpha(128)
        transparent.fill((170, 215, 81))
        self.screen.blit(transparent, (self.CELL_SIZE, self.CELL_SIZE))

        font = pygame.font.Font(self.font, 36)
        game_over_text = font.render(
            "Game Over",
            True,
            (87, 138, 52)
        )
        font = pygame.font.Font(self.font, 24)
        game_over_message_text = font.render(
            environment.game_over_message,
            True,
            (87, 138, 52)
        )

        # Combine the game over text and the game over message text
        # to center them on the screen
        x = (self.screen.get_width() - game_over_text.get_width()) / 2
        y = (self.screen.get_height() - (
            game_over_text.get_height() + game_over_message_text.get_height()
        )) / 2
        self.screen.blit(game_over_text, (x, y))
        x = (self.screen.get_width() - game_over_message_text.get_width()) / 2
        y += game_over_text.get_height()
        self.screen.blit(game_over_message_text, (x, y))
        pygame.display.flip()
        pygame.time.wait(500)

    def close(self):

        """
        Close the pygame window
        """

        pygame.quit()
        self._is_closed = True
        return False, None

    def disable(self):

        rectangle = pygame.Surface(
            (self.window_width, self.window_height)
        )
        rectangle.fill((87, 138, 52))
        self.screen.blit(rectangle, (0, 0))

        transparent = pygame.Surface(
            ((self.window_width / self.CELL_SIZE - 2) * self.CELL_SIZE,
             (self.window_height / self.CELL_SIZE - 2) * self.CELL_SIZE)
        )
        transparent.fill((170, 215, 81))
        self.screen.blit(transparent, (self.CELL_SIZE, self.CELL_SIZE))

        font = pygame.font.Font(self.font, 36)
        disabled_text = font.render(
            "GUI Disabled",
            True,
            (87, 138, 52)
        )

        font = pygame.font.Font(self.font, 24)
        message_text = font.render(
            "Press 'g' to enable the GUI",
            True,
            (87, 138, 52)
        )

        x = (self.screen.get_width() - disabled_text.get_width()) / 2
        y = (self.screen.get_height() - (
            disabled_text.get_height() + message_text.get_height()
        )) / 2
        self.screen.blit(disabled_text, (x, y))
        x = (self.screen.get_width() - message_text.get_width()) / 2
        y += disabled_text.get_height()
        self.screen.blit(message_text, (x, y))
        pygame.display.flip()

    def set_fps(self, fps: int):
        self.fps = fps
        self.clock.tick(self.fps)

    def is_closed(self) -> bool:
        """
        Return True if the pygame window is closed
        """
        return self._is_closed

    def lobby(self):

        # Display the lobby screen
        self.screen.fill((0, 0, 0))

        for y in range(self.board_height):
            for x in range(self.board_width):
                if (x + y) % 2 == 0:
                    cell_color = (170, 215, 81)
                else:
                    cell_color = (162, 209, 73)

                if (
                    x == 0 or x == self.board_width - 1 or
                    y == 0 or y == self.board_height - 1
                ):
                    cell_color = (87, 138, 52)

                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    (
                        x * self.CELL_SIZE, y * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                )

        # Display the title of the game
        font = pygame.font.Font(self.font, 36)
        title = font.render(
            "Learn2Slither",
            True,
            (87, 138, 52)
        )

        font = pygame.font.Font(self.font, 24)
        start = font.render(
            "Start",
            True,
            (87, 138, 52)
        )

        button_border = pygame.Surface((150, 50))
        button_border.fill((87, 138, 52))

        button = pygame.Surface((146, 46))
        button.fill((170, 215, 81))

        font = pygame.font.Font(self.font, 24)
        settings = font.render(
            "Settings",
            True,
            (87, 138, 52)
        )
        # x = (self.screen.get_width() - settings.get_width()) / 2
        # y += button.get_height() / 2 - settings.get_height() / 2
        # self.screen.blit(settings, (x, y))

        elements = [
            {
                "id": 0,
                "element": title,
                "x": 0,
                "y": 0
            },
            {
                "id": 1,
                "element": button_border,
                "x": 0,
                "y": button_border.get_height() + 10
            },
            {
                "id": 2,
                "element": button,
                "x": 2,
                "y": 2
            },
            {
                "id": 3,
                "element": start,
                "x": 0,
                "y": button_border.get_height() / 2 -
                1 - start.get_height() / 2
            },
            {
                "id": 4,
                "element": button_border,
                "x": 0,
                "y": button_border.get_height() + 10
            },
            {
                "id": 5,
                "element": button,
                "x": 2,
                "y": 2
            },
            {
                "id": 6,
                "element": settings,
                "x": 0,
                "y": button_border.get_height() / 2 - 1 -
                settings.get_height() / 2
            }
        ]

        elements_height = title.get_height() + \
            2 * (button_border.get_height() + 10)

        x = (self.screen.get_width() - title.get_width()) / 2
        y = (self.screen.get_height() - elements_height) / 2

        for element in elements:
            x = (self.screen.get_width() - element["element"].get_width()) / 2
            y += element["y"]
            self.screen.blit(element["element"], (x, y))

        pygame.display.flip()
        self.clock.tick(self.fps)
