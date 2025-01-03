import random
from Snake import Snake
from constants import RED_APPLE, GREEN_APPLE, WALL, EMPTY


class Environment:

    nb_red_apples = 0
    nb_green_apples = 1

    GRID_SIZE = 2
    width = height = GRID_SIZE + 2

    def __init__(self):

        self.nb_games = 0

        # Create the board with all cells set to 0 and walls around
        self.board = [
            [WALL if (
                x == 0 or x == self.width - 1 or
                y == 0 or y == self.height - 1
             ) else EMPTY
             for x in range(self.width)]
            for y in range(self.height)
        ]

        # Check if the number of apples + initial snake len is not too high
        if (
            self.nb_green_apples +
            self.nb_red_apples +
            Snake.initial_snake_length
            >
            (self.height - 2) * (self.width - 2)
        ):
            raise ValueError("Too many apples and/or initial snake length")

        # Set the snake on the board
        self.snake = Snake(self)

        # Set the green apples on the board
        for _ in range(self.nb_green_apples):
            self.new_apple(GREEN_APPLE)
        self.current_green_apples = self.nb_green_apples

        # Set the red apples on the board
        for _ in range(self.nb_red_apples):
            self.new_apple(RED_APPLE)
        self.current_red_apples = self.nb_red_apples

        self.is_game_over = False

    def reset(self):
        nb_games = self.nb_games
        self.__init__()
        self.nb_games = nb_games + 1

    def get_random_empty_cell(self):
        empty_cells = {
            (x, y)
            for x in range(1, self.width - 1)
            for y in range(1, self.height - 1)
            if self.board[y][x] == EMPTY
        }
        if not empty_cells:
            return None, None
        x, y = random.choice(list(empty_cells))
        return x, y

    def new_apple(self, apple):
        x, y = self.get_random_empty_cell()
        if x is not None and y is not None:
            self.board[y][x] = apple
        else:
            if apple == RED_APPLE:
                self.current_red_apples -= 1
                return self.game_over("Can't place a new red apple")
            elif apple == GREEN_APPLE:
                self.current_green_apples -= 1
                if self.current_green_apples == 0:
                    return self.game_over("No more green apples, you win!")

    def move_snake(self, direction):
        if direction in self.snake.directions:
            self.snake.direction = self.snake.directions[direction]
            is_game_over = self.snake.move(self)
            if is_game_over:
                self.game_over()

    def get_state(self):

        # Get the snake head position
        x_head, y_head = self.snake.body[0]

        # Create a 2d array with column and row, fill with spaces otherwise
        state = [
            [
                self[y][x] if x == x_head or y == y_head else ' '
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]

        return state

    def game_over(self, message=""):
        if message:
            print("GAME OVER:", message, "\n")
        self.is_game_over = True
        return True

    def is_running(self):
        return not self.is_game_over

    def __getitem__(self, key):
        return self.board[key]

    def __str__(self):
        state = self.get_state()
        res = "\033c"
        idx = 0
        for row in self.board:
            for cell in row:
                res += cell + " "
            res += "\t\t" + " ".join(state[idx]) + "\n"
            idx += 1
        return res
