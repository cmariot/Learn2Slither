import random
from constants import SNAKE_HEAD, SNAKE_BODY, RED_APPLE, GREEN_APPLE, WALL, EMPTY


class Snake:

    initial_snake_length = 3

    directions = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }

    def __init__(self, board):

        all_positions = {
            (x, y)
            for x in range(1, board.width - 1)
            for y in range(1, board.height - 1)
        }

        while True:

            # Choose a random head position
            head_position = random.choice(list(all_positions))
            body = [head_position]

            # Remove the head position from the available positions
            available_positions = all_positions - {head_position}

            # Set the body of the snake
            for _ in range(self.initial_snake_length - 1):
                x, y = body[-1]
                possible_positions = [
                    (x + dir_x, y + dir_y)
                    for dir_x, dir_y in list(self.directions.values())
                ]
                possible_positions = [
                    pos
                    for pos in possible_positions
                    if pos in available_positions
                ]
                if not possible_positions:
                    break
                new_position = random.choice(possible_positions)
                body.append(new_position)
                available_positions.remove(new_position)

            if len(body) == self.initial_snake_length:
                break

        self.body = body
        self.direction = random.choice(list(self.directions.values()))

        # Set the snake on the board
        for x, y in self.body:
            board.board[y][x] = SNAKE_BODY
        x, y = self.body[0]
        board.board[y][x] = SNAKE_HEAD

    def move(self, board):
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        x, y = new_head
        next_cell = board[y][x]
        if next_cell == WALL:
            raise ValueError("Snake hit the wall")
        elif next_cell == SNAKE_BODY:
            raise ValueError("Snake collision")
        elif next_cell == RED_APPLE:
            self.shrink(board, new_head, x, y)
        elif next_cell == GREEN_APPLE:
            self.grow(board, new_head, x, y)
        else:
            self.move_forward(board, new_head, x, y)

    def move_forward(self, board, new_head, x, y):
        board[y][x] = SNAKE_HEAD
        x, y = self.body[0]
        board[y][x] = SNAKE_BODY
        x, y = self.body[-1]
        board[y][x] = EMPTY
        self.body = [new_head] + self.body[:-1]

    def grow(self, board, new_head, x, y):
        # Green apple : grow the snake and add a new Green apple
        board[y][x] = SNAKE_HEAD
        x, y = self.body[0]
        board[y][x] = SNAKE_BODY
        self.body = [new_head] + self.body
        board.new_apple(GREEN_APPLE)

    def shrink(self, board, new_head, x, y):
        board[y][x] = SNAKE_HEAD
        if len(self.body) == 1:
            self.die()
        if len(self.body) > 2:
            x, y = self.body[0]
            board[y][x] = SNAKE_BODY
        x, y = self.body[-2]
        board[y][x] = EMPTY
        x, y = self.body[-1]
        board[y][x] = EMPTY
        self.body = [new_head] + self.body[:-2]
        board.new_apple(RED_APPLE)

    def win(self):
        print("You win")
        exit(0)

    def die(self):
        raise ValueError("Snake died")
