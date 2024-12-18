from Environment import Environment
from game_mode import GameMode
from constants import RED_APPLE, GREEN_APPLE, SNAKE_HEAD, SNAKE_BODY, WALL
import pygame


class GraphicalUserInteface:

    FPS = 10
    CELL_SIZE = 40

    def __init__(self, environment: Environment):
        pygame.init()
        self.clock = pygame.time.Clock()
        window_width = environment.width * self.CELL_SIZE
        window_height = environment.height * self.CELL_SIZE
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Learn2Slither")

    def draw(self, environment: Environment):
        self.screen.fill((0, 0, 0))
        for y, row in enumerate(environment.board):
            for x, cell in enumerate(row):
                cell = environment.board[y][x]
                if cell == RED_APPLE:
                    cell_color = (255, 0, 0)
                elif cell == GREEN_APPLE:
                    cell_color = (0, 255, 0)
                elif cell == SNAKE_HEAD:
                    cell_color = (0, 0, 255)
                elif cell == SNAKE_BODY:
                    cell_color = (150, 150, 255)
                elif cell == WALL:
                    cell_color = (100, 100, 100)
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
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def handle_key_pressed(self, environment: Environment, game_mode: GameMode):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key == 'space':
                    game_mode.switch()
                elif key == 'q' or key == 'escape':
                    self.close()

                if game_mode.is_human():
                    if key in ("up", "down", "left", "right"):
                        environment.move_snake(key)

    def close(self):
        pygame.quit()
        exit()
