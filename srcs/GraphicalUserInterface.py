from Environment import Environment
from game_mode import GameMode
from constants import RED_APPLE, GREEN_APPLE, SNAKE_HEAD, SNAKE_BODY, WALL
import pygame


class GraphicalUserInteface:

    FPS = 20
    CELL_SIZE = 40

    def __init__(self, environment: Environment):
        pygame.init()
        self.clock = pygame.time.Clock()
        window_width = environment.width * self.CELL_SIZE
        window_height = environment.height * self.CELL_SIZE
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.load_snake_images()
        self.load_apple_images()
        pygame.display.set_caption("Learn2Slither")

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

    def draw(self, environment: Environment):
        self.screen.fill((0, 0, 0))
        for y, row in enumerate(environment.board):
            for x, cell in enumerate(row):
                cell = environment.board[y][x]
                if (x + y) % 2 == 0:
                    cell_color = (235, 235, 235)
                else:
                    cell_color = (255, 255, 255)
                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    (
                        x * self.CELL_SIZE, y * self.CELL_SIZE,
                        self.CELL_SIZE - 1, self.CELL_SIZE - 1
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
                        (x * self.CELL_SIZE + 0.25 * self.CELL_SIZE, y * self.CELL_SIZE + 0.25 * self.CELL_SIZE)
                    )
                elif cell == GREEN_APPLE:
                    self.screen.blit(
                        self.green_apple,
                        (x * self.CELL_SIZE + 0.25 * self.CELL_SIZE, y * self.CELL_SIZE + 0.25 * self.CELL_SIZE)
                    )

                else:
                    if cell == WALL:
                        cell_color = (50, 50, 50)
                    pygame.draw.rect(
                        self.screen,
                        cell_color,
                        (
                            x * self.CELL_SIZE, y * self.CELL_SIZE,
                            self.CELL_SIZE - 1, self.CELL_SIZE - 1
                        )
                    )
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def handle_key_pressed(self, environment: Environment, game_mode: GameMode):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key == 'space':
                    game_mode.switch()
                elif key == 'q' or key == 'escape':
                    self.close()
                elif game_mode.is_human():
                    if key in ("up", "down", "left", "right"):
                        environment.move_snake(key)
        return True

    def close(self):
        pygame.quit()
        exit()
