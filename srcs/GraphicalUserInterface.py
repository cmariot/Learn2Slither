from Environment import Environment
from constants import RED_APPLE, GREEN_APPLE, SNAKE_HEAD, SNAKE_BODY, WALL
import pygame
from InterfaceController import InterfaceController


class GraphicalUserInteface:

    CELL_SIZE = 40

    def __init__(self, board_width, board_height, fps):
        pygame.init()
        self.clock = pygame.time.Clock()
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
        self.is_closed = False
        self.fps = fps

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
                gui, cli,
                score_evolution
            ):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.close(environment)
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)

                print(f"{key} pressed")

                if (
                    key in ('return', 'enter') and
                    controller.is_ai() and
                    controller.step_by_step
                ):
                    return True, None
                elif key == 'space':
                    controller.toggle_ai()
                elif key == 'c':
                    controller.toggle_cli()
                elif key == 'g':
                    controller.toggle_gui(
                        gui, environment, score_evolution
                    )
                elif key == 'q' or key == 'escape':
                    return self.close(environment)
                elif key in ('[+]', '[-]'):
                    controller.change_fps(key, gui, cli)
                elif controller.is_human():
                    if key in ("up", "down", "left", "right"):
                        key = ('up', 'down', 'left', 'right').index(key)
                        return True, key

        if controller.step_by_step and controller.is_ai():
            return False, None
        return controller.is_ai(), None

    def draw(self, environment, scores, controller: InterfaceController):

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
                        if direction == (0, -1):
                            image = self.snake_head_up
                        elif direction == (0, 1):
                            image = self.snake_head_down
                        elif direction == (-1, 0):
                            image = self.snake_head_left
                        else:
                            image = self.snake_head_right
                        self.screen.blit(
                            image,
                            (x * self.CELL_SIZE, y * self.CELL_SIZE)
                        )

                    else:

                        snake_length = environment.snake.get_snake_length()
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
            f"Score: {scores.score}",
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
        TEST_WIDTH = high_score_text.get_width()
        x = WINDOW_WIDTH - TEST_WIDTH - 10
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

    def close(self, environment: Environment):
        pygame.quit()
        environment.is_closed = True
        self.is_closed = True
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

    def set_fps(self, fps):
        self.fps = fps
        self.clock.tick(self.fps)
