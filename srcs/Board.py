import random


class Board:

    height = 10
    width = 10

    nb_green_apples = 2
    nb_red_apples = 1

    def __init__(self):

        # Create the board with all cells set to 0
        self.board = [
            [0 for _ in range(self.width)]
            for _ in range(self.height)
        ]

        # Check if the number of apples is not too high
        if self.nb_green_apples + self.nb_red_apples > self.height * self.width:
            raise ValueError("Too many green apples")

        # Set the green apples on the board (value 'G')
        for _ in range(self.nb_green_apples):
            GREEN = '\033[32m'
            RESET = '\033[0m'
            self.new_apple(f'{GREEN}G{RESET}')

        # Set the red apples on the board (value 'R')
        for _ in range(self.nb_red_apples):
            RED = '\033[31m'
            RC = '\033[31m'
            self.new_apple(f'{RED}R{RC}')

    def new_apple(self, color='G'):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.board[y][x] == 0:
                break
        self.board[y][x] = color

    def __str__(self):
        return "\n".join([" ".join([str(cell) for cell in row]) for row in self.board])