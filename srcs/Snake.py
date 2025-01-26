import random
from constants import (
    SNAKE_HEAD, SNAKE_BODY, RED_APPLE, GREEN_APPLE, WALL, EMPTY,
    NEGATIVE_REWARD, POSITIVE_REWARD, SMALLLER_NEGATIVE_REWARD,
    BIGGER_NEGATIVE_REWARD
)
from Directions import Directions


random.seed(0)


class Snake:

    initial_snake_length = 3
    directions = Directions().get_directions()

    def __init__(self, board):

        all_positions = {
            (x, y)
            for x in range(1, board.width - 1)
            for y in range(1, board.height - 1)
        }

        is_first_body_part = True
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

                if is_first_body_part:

                    # Determine the direction of the snake
                    if len(body) > 1:
                        x_head, y_head = body[0]
                        x_next, y_next = body[1]
                        self.direction = (x_head - x_next, y_head - y_next)
                    else:
                        self.direction = random.choice(
                            list(self.directions.values())
                        )

                    is_first_body_part = False
                available_positions.remove(new_position)

            if len(body) == self.initial_snake_length:
                break

        self.body = body
        self.board_width = board.width - 2

        # Set the snake on the board
        for x, y in self.body:
            board.board[x][y] = SNAKE_BODY
        x, y = self.get_head_position()
        board.board[x][y] = SNAKE_HEAD

    def move(self, board):
        head_x, head_y = self.get_head_position()
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        next_cell = board[new_head[0]][new_head[1]]
        if next_cell == WALL:
            return self.die("Snake hit the wall")
        elif next_cell == SNAKE_BODY:
            return self.die("Snake collision")
        elif next_cell == RED_APPLE:
            return self.shrink(board, new_head)
        elif next_cell == GREEN_APPLE:
            return self.grow(board, new_head)
        elif next_cell == EMPTY:
            game_over, reward, game_over_msg = \
                  self.move_forward(board, new_head)
            # If the snake is moving in a GREEN_APPLE direction POSITIVE_REWARD
            green_apple_distance = self.board_width
            wall_distance = 0
            for i in range(1, self.board_width):
                next_cell = board[head_x + i * dir_x][head_y + i * dir_y]
                if next_cell == WALL:
                    wall_distance = i - 1
                    break
                elif next_cell == GREEN_APPLE:
                    green_apple_distance = i - 1
            if green_apple_distance < wall_distance:
                reward *= -1
            return game_over, reward, game_over_msg

    def move_forward(self, board, new_head):
        x, y = new_head
        board[x][y] = SNAKE_HEAD
        x, y = self.get_head_position()
        board[x][y] = SNAKE_BODY
        x, y = self.body[-1]
        board[x][y] = EMPTY
        self.body = [new_head] + self.body[:-1]
        return False, SMALLLER_NEGATIVE_REWARD, "Snake moved forward"

    def grow(self, board, new_head):
        # Green apple : grow the snake and add a new Green apple
        x, y = new_head
        board[x][y] = SNAKE_HEAD
        x, y = self.get_head_position()
        board[x][y] = SNAKE_BODY
        self.body = [new_head] + self.body
        board.new_apple(GREEN_APPLE)
        return False, POSITIVE_REWARD, "Snake grew up by eating a green apple"

    def shrink(self, board, new_head):
        x, y = new_head
        if len(self.body) == 1:
            return self.die("Snake has no more body")
        board[x][y] = SNAKE_HEAD
        if len(self.body) > 2:
            x, y = self.get_head_position()
            board[x][y] = SNAKE_BODY
        x, y = self.body[-2]
        board[x][y] = EMPTY
        x, y = self.body[-1]
        board[x][y] = EMPTY
        self.body = [new_head] + self.body[:-2]
        board.new_apple(RED_APPLE)
        return False, NEGATIVE_REWARD, "Snake shrunk by eating a red apple"

    def die(self, message):
        return True, BIGGER_NEGATIVE_REWARD, message

    def get_body_index(self, x, y):
        """
        This method returns the index of the body part at position (x, y).
        """
        for index, (body_x, body_y) in enumerate(self.body):
            if body_x == x and body_y == y:
                return index
        return -1

    def len(self):
        return len(self.body)

    def get_head_position(self):
        return self.body[0]
