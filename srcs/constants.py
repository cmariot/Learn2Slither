# Colors for the print function
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
GRAY = '\033[90m'
RESET = '\033[0m'

# Clear the terminal
CLEAR = '\033[H\033[J'

# Constants for the board
GREEN_APPLE = GREEN + 'G' + RESET
RED_APPLE = RED + 'R' + RESET
SNAKE_HEAD = BLUE + 'H' + RESET
SNAKE_BODY = BLUE + 'S' + RESET
EMPTY = '0'
WALL = GRAY + 'W' + RESET
DEAD_SNAKE = GRAY + 'X' + RESET

# Constants for the rewards
NEGATIVE_REWARD = -10  # Manger une pomme rouge
POSITIVE_REWARD = 150  # Manger une pomme verte
SMALLLER_NEGATIVE_REWARD = -1  # Se deplacer
BIGGER_NEGATIVE_REWARD = -100  # Collision

# Constants for the training and gaming loops
TRAINING_LOOP = True
GAMING_LOOP = True
