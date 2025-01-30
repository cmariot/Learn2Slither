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
        self.state = "MENU" if not args.no_gui else "GAME"
        self.menu = Menu(self)
        self.game = Game(self)
        self.settings = Settings(self)
        self.help = Help(self)

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

    def close(self):

        """
        Close the pygame window
        """

        pygame.quit()
        self._is_closed = True
        return False, None

    def is_closed(self) -> bool:
        """
        Return True if the pygame window is closed
        """
        return self._is_closed

    def set_fps(self, fps: int):
        self.fps = fps
        self.clock.tick(self.fps)


class Menu:

    def __init__(self, gui):
        self.gui = gui

    def draw_menu(self):
        self.gui.screen.fill((0, 0, 0))
        for y in range(self.gui.board_height):
            for x in range(self.gui.board_width):
                if (x + y) % 2 == 0:
                    cell_color = (170, 215, 81)
                else:
                    cell_color = (162, 209, 73)

                if (
                    x == 0 or x == self.gui.board_width - 1 or
                    y == 0 or y == self.gui.board_height - 1
                ):
                    cell_color = (87, 138, 52)

                pygame.draw.rect(
                    self.gui.screen,
                    cell_color,
                    (
                        x * self.gui.CELL_SIZE, y * self.gui.CELL_SIZE,
                        self.gui.CELL_SIZE, self.gui.CELL_SIZE
                    )
                )
        font = pygame.font.Font(self.gui.font, 36)
        title = font.render("Learn2Slither", True, (87, 138, 52))
        self.gui.screen.blit(
            title, (self.gui.window_width // 2 - title.get_width() // 2, 100))

        title_end_y = 100 + title.get_height()

        button_height = 50

        self.menu_start_button = Button(
            self.gui.window_width // 2 - 75,
            title_end_y + 10 + button_height,
            150, 50, "Start", (87, 138, 52), self.gui.screen
        )

        button_end_y = title_end_y + 10 + self.menu_start_button.height + 10

        self.menu_settings_button = Button(
            self.gui.window_width // 2 - 75,
            button_end_y + 10 + button_height,
            150, 50, "Settings", (87, 138, 52), self.gui.screen
        )

        button_end_y += 10 + self.menu_settings_button.height + 10

        self.menu_quit_button = Button(
            self.gui.window_width // 2 - 75,
            button_end_y + 10 + button_height,
            150, 50, "Exit", (87, 138, 52), self.gui.screen
        )

        pygame.display.flip()

    def handle_menu_events(self, gui: GraphicalUserInterface):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gui.close()
                gui.state = "EXIT"
                return "EXIT"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if self.menu_start_button.is_clicked(x, y):
                    self.gui.state = "GAME"
                elif self.menu_settings_button.is_clicked(x, y):
                    self.gui.state = "SETTINGS"
                elif self.menu_quit_button.is_clicked(x, y):
                    return "EXIT"
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key == "q" or key == "escape":
                    self.gui.state = "EXIT"
                    return "EXIT"
                elif key == "space":
                    self.gui.state = "GAME"
        return None


class Game:

    def __init__(self, gui):
        self.gui = gui

    def draw(
            self,
            environment: Environment,
            scores: Score,
            controller: InterfaceController
    ):

        if controller.gui_disabled():
            return

        self.gui.screen.fill((0, 0, 0))
        for y, row in enumerate(environment.board):
            for x, cell in enumerate(row):
                cell = environment.board[x][y]
                if (x + y) % 2 == 0:
                    cell_color = (170, 215, 81)
                else:
                    cell_color = (162, 209, 73)
                pygame.draw.rect(
                    self.gui.screen,
                    cell_color,
                    (
                        x * self.gui.CELL_SIZE, y * self.gui.CELL_SIZE,
                        self.gui.CELL_SIZE, self.gui.CELL_SIZE
                    )
                )
                if cell == SNAKE_HEAD or cell == SNAKE_BODY:
                    if cell == SNAKE_HEAD:
                        direction = environment.snake.direction
                        image = {
                            (0, -1): self.gui.snake_head_up,
                            (0, 1): self.gui.snake_head_down,
                            (-1, 0): self.gui.snake_head_left,
                            (1, 0): self.gui.snake_head_right
                        }[direction]
                        self.gui.screen.blit(
                            image,
                            (x * self.gui.CELL_SIZE, y * self.gui.CELL_SIZE)
                        )
                    else:
                        snake_length = environment.snake.len()
                        body_index = environment.snake.get_body_index(x, y)
                        if body_index == snake_length - 1:
                            (
                                previous_body_x, previous_body_y
                            ) = environment.snake.body[-2]
                            if previous_body_x == x and previous_body_y < y:
                                image = self.gui.snake_tail_up
                            elif previous_body_x == x and previous_body_y > y:
                                image = self.gui.snake_tail_down
                            elif previous_body_x < x and previous_body_y == y:
                                image = self.gui.snake_tail_left
                            else:
                                image = self.gui.snake_tail_right
                            self.gui.screen.blit(
                                image,
                                (x * self.gui.CELL_SIZE,
                                 y * self.gui.CELL_SIZE)
                            )
                        else:
                            (
                                previous_body_x, previous_body_y
                            ) = environment.snake.body[body_index + 1]
                            (
                                next_body_x, next_body_y
                            ) = environment.snake.body[body_index - 1]
                            if (
                                previous_body_x < x and
                                next_body_x > x or previous_body_x > x and
                                next_body_x < x
                            ):
                                image = self.gui.body_horizontal
                            elif (
                                previous_body_y < y and
                                next_body_y > y or
                                previous_body_y > y and
                                next_body_y < y
                            ):
                                image = self.gui.body_vertical
                            elif (
                                previous_body_x < x and
                                next_body_y < y or
                                previous_body_y < y and
                                next_body_x < x
                            ):
                                image = self.gui.body_top_right
                            elif (
                                previous_body_x > x and
                                next_body_y < y or
                                previous_body_y < y and
                                next_body_x > x
                            ):
                                image = self.gui.body_top_left
                            elif (
                                previous_body_x < x and
                                next_body_y > y or
                                previous_body_y > y and
                                next_body_x < x
                            ):
                                image = self.gui.body_bottom_right
                            else:
                                image = self.gui.body_bottom_left
                            self.gui.screen.blit(
                                image,
                                (x * self.gui.CELL_SIZE,
                                 y * self.gui.CELL_SIZE)
                            )
                elif cell == RED_APPLE:
                    self.gui.screen.blit(
                        self.gui.red_apple,
                        (
                            x * self.gui.CELL_SIZE + 0.25 * self.gui.CELL_SIZE,
                            y * self.gui.CELL_SIZE + 0.25 * self.gui.CELL_SIZE
                        )
                    )
                elif cell == GREEN_APPLE:
                    self.gui.screen.blit(
                        self.gui.green_apple,
                        (
                            x * self.gui.CELL_SIZE + 0.25 * self.gui.CELL_SIZE,
                            y * self.gui.CELL_SIZE + 0.25 * self.gui.CELL_SIZE
                        )
                    )
                elif cell == WALL:
                    pygame.draw.rect(
                        self.gui.screen,
                        (87, 138, 52),
                        (
                            x * self.gui.CELL_SIZE, y * self.gui.CELL_SIZE,
                            self.gui.CELL_SIZE, self.gui.CELL_SIZE
                        )
                    )

        WINDOW_WIDTH = WINDOW_HEIGHT = environment.width * self.gui.CELL_SIZE

        # Display the score on the screen
        font = pygame.font.Font(self.gui.font, 24)
        score_text = font.render(
            f"Snake len: {scores.snake_len}",
            True,
            (170, 215, 81)
        )
        x = 10
        y = WINDOW_HEIGHT - self.gui.CELL_SIZE + \
            (self.gui.CELL_SIZE - score_text.get_height()) / 2
        self.gui.screen.blit(score_text, (x, y))

        max_snake_len = max(scores.max_snake_len, max(
            scores.max_snake_lengths[scores.initial_game_number:],
            default=scores.max_snake_len
        ))
        high_score_text = font.render(
            f"Max: {max_snake_len}",
            True,
            (170, 215, 81)
        )
        x = WINDOW_WIDTH - high_score_text.get_width() - 10
        self.gui.screen.blit(high_score_text, (x, y))

        # Game number
        game_number_text = font.render(
            f"Game #{scores.game_number}",
            True,
            (170, 215, 81)
        )
        x = (WINDOW_WIDTH - game_number_text.get_width()) / 2
        y = (self.gui.CELL_SIZE - game_number_text.get_height()) / 2
        self.gui.screen.blit(game_number_text, (x, y))

        # Display the FPS on the screen
        font = pygame.font.Font(self.gui.font, 12)
        fps = int(self.gui.clock.get_fps())
        fps_text = font.render(
            f"FPS: {fps} / {self.gui.fps}",
            True,
            (170, 215, 81)
        )
        x = 10
        y = (self.gui.CELL_SIZE - fps_text.get_height()) / 2
        self.gui.screen.blit(fps_text, (x, y))

        # Key mapping button
        self.help_button = Button(
            WINDOW_WIDTH - 70,
            (self.gui.CELL_SIZE - game_number_text.get_height()) / 2,
            60,
            game_number_text.get_height(),
            "Help", (87, 138, 52),
            self.gui.screen, 12
        )

        pygame.display.flip()
        self.gui.clock.tick(self.gui.fps)

    def handle_key_pressed(
        self,
        environment: Environment,
        controller: InterfaceController,
        cli: CommandLineInterface,
        scores: Score,
        agent: Agent,
        gui: GraphicalUserInterface
    ) -> tuple[bool, int]:

        """
        Handle the key pressed by the user.
        Return a tuple with a boolean to indicate if the move should be
        performed and the action to perform as an integer.
        """

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return self.gui.close()

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
                    controller.toggle_gui(self.gui, environment, scores)

                elif key == 'p':
                    # Enable/disable the step by step mode
                    controller.toggle_step_by_step()

                elif key == 'q' or key == 'escape':
                    return self.gui.close()

                elif key in ('[+]', '[-]', '-', '='):
                    # Increase or decrease the FPS (+/- 10 fps)
                    controller.change_fps(key, gui, cli)

                elif key == 's':
                    # Save the model and the score evolution
                    agent.save(scores)

                elif controller.is_human():
                    # Handle the key pressed by the user
                    if key in ("up", "down", "left", "right"):
                        key = ('up', 'down', 'left', 'right').index(key)
                        agent.action = key
                        return True, key

            elif event.type == pygame.MOUSEBUTTONDOWN:

                # Click on the help button
                x, y = pygame.mouse.get_pos()

                if self.help_button.is_clicked(x, y):
                    if controller.gui_disabled():
                        return False, None
                    controller.toggle_help()
                    gui.state = "HELP"
                    return False, None

        if controller.step_by_step and controller.is_ai():
            return False, None

        return controller.is_ai(), None

    def game_over(
            self,
            environment: Environment,
            controller: InterfaceController,
            gui: GraphicalUserInterface
    ):

        if controller.gui_disabled():
            return

        # Display a transparent rectangle on the screen
        transparent = pygame.Surface(
            ((environment.width - 2) * self.gui.CELL_SIZE,
             (environment.height - 2) * self.gui.CELL_SIZE)
        )
        transparent.set_alpha(128)
        transparent.fill((170, 215, 81))
        gui.screen.blit(transparent, (self.gui.CELL_SIZE, self.gui.CELL_SIZE))

        font = pygame.font.Font(gui.font, 36)
        game_over_text = font.render(
            "Game Over",
            True,
            (87, 138, 52)
        )
        font = pygame.font.Font(gui.font, 24)
        game_over_message_text = font.render(
            environment.game_over_message,
            True,
            (87, 138, 52)
        )

        # Combine the game over text and the game over message text
        # to center them on the screen
        x = (gui.screen.get_width() - game_over_text.get_width()) / 2
        y = (gui.screen.get_height() - (
            game_over_text.get_height() + game_over_message_text.get_height()
        )) / 2
        gui.screen.blit(game_over_text, (x, y))
        x = (gui.screen.get_width() - game_over_message_text.get_width()) / 2
        y += game_over_text.get_height()
        gui.screen.blit(game_over_message_text, (x, y))
        pygame.display.flip()
        pygame.time.wait(500)

    def disable(self, gui: GraphicalUserInterface):

        """
        When the GUI is disabled in game, display a message on the screen
        """

        # green rectangle on the screen
        rectangle = pygame.Surface(
            (gui.window_width, gui.window_height)
        )
        rectangle.fill((87, 138, 52))
        gui.screen.blit(rectangle, (0, 0))

        # transparent rectangle
        transparent = pygame.Surface(
            ((gui.window_width / gui.CELL_SIZE - 2) * gui.CELL_SIZE,
             (gui.window_height / gui.CELL_SIZE - 2) * gui.CELL_SIZE)
        )
        transparent.fill((170, 215, 81))
        gui.screen.blit(transparent, (gui.CELL_SIZE, gui.CELL_SIZE))

        # Text
        font = pygame.font.Font(gui.font, 36)
        disabled_text = font.render(
            "GUI Disabled",
            True,
            (87, 138, 52)
        )
        font = pygame.font.Font(gui.font, 24)
        message_text = font.render(
            "Press 'g' to enable the GUI",
            True,
            (87, 138, 52)
        )
        x = (gui.screen.get_width() - disabled_text.get_width()) / 2
        y = (gui.screen.get_height() - (
            disabled_text.get_height() + message_text.get_height()
        )) / 2
        gui.screen.blit(disabled_text, (x, y))
        x = (gui.screen.get_width() - message_text.get_width()) / 2
        y += disabled_text.get_height()
        gui.screen.blit(message_text, (x, y))

        pygame.display.flip()


class Text:

    def __init__(self, x, y, text, text_color, font, screen):
        self.x = x
        self.y = y
        self.text = text
        self.text_color = text_color
        self.font = font
        self.width, self.height = self.font.size(self.text)
        self.draw(screen)

    def draw(self, screen):
        label = self.font.render(self.text, True, self.text_color)
        screen.blit(
            label,
            (self.x + (self.width - label.get_width()) // 2, self.y +
             (self.height - label.get_height()) // 2)
        )


class Button:

    def __init__(
            self, x, y, width, height, text, text_color, screen, font_size=24
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.text_color = text_color
        self.font_size = font_size
        self.draw(screen)

    def draw(self, screen):
        button_border = pygame.Surface((self.width, self.height))
        button_border.fill((87, 138, 52))
        button = pygame.Surface((self.width - 4, self.height - 4))
        button.fill((170, 215, 81))
        screen.blit(button_border, (self.x, self.y))
        screen.blit(button, (self.x + 2, self.y + 2))

        font = pygame.font.Font(pygame.font.get_default_font(), self.font_size)
        label = font.render(self.text, True, self.text_color)
        screen.blit(
            label,
            (self.x + (self.width - label.get_width()) // 2, self.y +
             (self.height - label.get_height()) // 2)
        )

    def is_clicked(self, x, y):
        return self.x <= x <= self.x + self.width and \
              self.y <= y <= self.y + self.height


class Rectangle:

    def __init__(self, x, y, width, height, color, screen):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.draw(screen)

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (self.x, self.y, self.width, self.height)
        )


class Settings:

    def __init__(self, gui):
        self.gui = gui

    def draw(self, gui: GraphicalUserInterface):
        self.gui.screen.fill((0, 0, 0))
        for y in range(self.gui.board_height):
            for x in range(self.gui.board_width):
                if (x + y) % 2 == 0:
                    cell_color = (170, 215, 81)
                else:
                    cell_color = (162, 209, 73)

                if (
                    x == 0 or x == self.gui.board_width - 1 or
                    y == 0 or y == self.gui.board_height - 1
                ):
                    cell_color = (87, 138, 52)

                pygame.draw.rect(
                    self.gui.screen,
                    cell_color,
                    (
                        x * self.gui.CELL_SIZE, y * self.gui.CELL_SIZE,
                        self.gui.CELL_SIZE, self.gui.CELL_SIZE
                    )
                )
        font = pygame.font.Font(self.gui.font, 36)
        title = font.render("Settings", True, (87, 138, 52))

        x = self.gui.window_width // 2 - title.get_width() // 2
        y = 100
        self.gui.screen.blit(title, (x, y))

        # FPS Settings
        y += title.get_height() + 10
        x = (self.gui.window_width - 168) // 2
        self.settings_fps_text = Text(
            x, y,
            f"FPS: {gui.fps}", (87, 138, 52),
            pygame.font.Font(self.gui.font, 24), self.gui.screen
        )

        # Decrease FPS button
        x += self.settings_fps_text.width + 20
        self.settings_decrease_fps_button = Button(
            x, y, self.settings_fps_text.height, self.settings_fps_text.height,
            "-", (87, 138, 52), self.gui.screen, 12
        )

        # Increase FPS button
        x += self.settings_decrease_fps_button.width + 10
        self.settings_increase_fps_button = Button(
            x, y, self.settings_fps_text.height, self.settings_fps_text.height,
            "+", (87, 138, 52), self.gui.screen, 12
        )

        # Back button
        x = self.gui.window_width // 2 - 75
        y += self.settings_fps_text.height + 20
        self.settings_back_button = Button(
            x, y, 150, 50, "Back", (87, 138, 52), self.gui.screen, 18
        )

        pygame.display.flip()

    def handle_key_pressed(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.gui.close()
                return "EXIT"

            elif event.type == pygame.KEYDOWN:

                key = pygame.key.name(event.key)

                if key == 'q' or key == 'escape':
                    self.gui.close()
                    return "EXIT"
                elif key in ('[+]', '='):
                    return "FPS+"
                elif key in ('-', '[-]'):
                    return "FPS-"

            # Handle the mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if self.settings_increase_fps_button.is_clicked(x, y):
                    return "FPS+"
                elif self.settings_decrease_fps_button.is_clicked(x, y):
                    return "FPS-"
                elif self.settings_back_button.is_clicked(x, y):
                    self.gui.state = "MENU"
                    return "BACK"

        return None


class Help:

    def __init__(self, gui):
        self.gui = gui

    def draw(self):

        """
        Display the list of key mappings of the game
        """

        key_mapping = {
            "Toggle AI/Human mode": "Space",
            "In AI mode": "The agent plays",
            "In Human mode": "Arrow keys to move",
            "Toggle CLI": "C",
            "Toggle GUI": "G",
            "Step by step": "P",
            "In step by step mode": "Enter to perform the next move",
            "Quit": "Q or Esc",
            "Save the model": "S",
            "Increase/Decrease FPS": "+/-",
        }

        self.gui.screen.fill((0, 0, 0))

        for y in range(self.gui.board_height):
            for x in range(self.gui.board_width):
                if (x + y) % 2 == 0:
                    cell_color = (170, 215, 81)
                else:
                    cell_color = (162, 209, 73)

                if (
                    x == 0 or x == self.gui.board_width - 1 or
                    y == 0 or y == self.gui.board_height - 1
                ):
                    cell_color = (87, 138, 52)

                pygame.draw.rect(
                    self.gui.screen,
                    cell_color,
                    (
                        x * self.gui.CELL_SIZE, y * self.gui.CELL_SIZE,
                        self.gui.CELL_SIZE, self.gui.CELL_SIZE
                    )
                )

        font = pygame.font.Font(self.gui.font, 36)
        title = font.render("Help", True, (87, 138, 52))
        x = self.gui.window_width // 2 - title.get_width() // 2
        y = 50
        self.gui.screen.blit(title, (x, y))

        font = pygame.font.Font(self.gui.font, 14)

        max_len = 0
        max_text = ""
        for key, value in key_mapping.items():
            if len(key) + len(value) > max_len:
                max_text = f"{key}: {value}"
                max_len = len(max_text)

        key_text = font.render(
            max_text,
            True,
            (87, 138, 52)
        )

        x = self.gui.window_width // 2 - key_text.get_width() // 2
        y += title.get_height() + 10

        for key, value in key_mapping.items():

            key_text = font.render(
                f"{key}: {value}",
                True,
                (87, 138, 52)
            )
            self.gui.screen.blit(key_text, (x, y))
            y += key_text.get_height() + 10

        # Back button
        x = self.gui.window_width // 2 - 75
        y = self.gui.window_height - 100
        self.help_back_button = Button(
            self.gui.board_width * self.gui.CELL_SIZE - 70,
            (self.gui.CELL_SIZE - 24) / 2,
            60,
            24,
            "Back", (87, 138, 52),
            self.gui.screen, 12
        )

        # Return to the menu button (right top corner)
        self.help_home_button = Button(
            x, y, 150, 50, "Return home", (87, 138, 52), self.gui.screen, 18
        )

        pygame.display.flip()

    def handle_key_pressed(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.gui.close()
                return "EXIT"

            elif event.type == pygame.KEYDOWN:

                key = pygame.key.name(event.key)

                if key == 'q' or key == 'escape':
                    self.gui.close()
                    return "EXIT"

            # Handle the mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if self.help_back_button.is_clicked(x, y):
                    return "BACK"
                elif self.help_home_button.is_clicked(x, y):
                    return "HOME"

        return None
