import random
import pygame


RED = '\033[31m'
GREEN = '\033[32m'
PURPLE = '\033[35m'
RESET = '\033[0m'

class Board:

    height = 10
    width = 10

    nb_green_apples = 2
    nb_red_apples = 1

    initial_snake_length = 3

    def __init__(self):

        # Create the board with all cells set to 0
        self.board = [
            [0 for _ in range(self.width)]
            for _ in range(self.height)
        ]

        # Check if the number of apples is not too high
        if (
            self.nb_green_apples +
            self.nb_red_apples +
            self.initial_snake_length
            >
            self.height * self.width
        ):
            raise ValueError("Too many green apples")

        # Set the green apples on the board (value 'G')
        for _ in range(self.nb_green_apples):
            self.new_apple(f'{GREEN}G{RESET}')

        # Set the red apples on the board (value 'R')
        for _ in range(self.nb_red_apples):
            self.new_apple(f'{RED}R{RESET}')

        # Set the snake on the board (value 'S')
        self.set_snake()

    def new_apple(self, color='G'):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.board[y][x] == 0:
                break
        self.board[y][x] = color

    def set_snake(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.board[y][x] == 0:
                break
        self.board[y][x] = f'{PURPLE}H{RESET}'
        self.snake = [(x, y)]
        for _ in range(self.initial_snake_length - 1):
            while True:
                # Random betweem 0 and 3, next to the last cell of the snake
                if random.randint(0, 1):
                    x = self.snake[-1][0] + random.choice([-1, 1])
                    y = self.snake[-1][1]
                else:
                    x = self.snake[-1][0]
                    y = self.snake[-1][1] + random.choice([-1, 1])
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    continue
                if self.board[y][x] == 0:
                    break
            self.board[y][x] = f'{PURPLE}S{RESET}'
            self.snake.append((x, y))


    def gui(self):
        pygame.init()


    def __str__(self):
        return "\n".join([" ".join([str(cell) for cell in row]) for row in self.board])