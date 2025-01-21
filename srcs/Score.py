import matplotlib.pyplot as plt
import pandas
import os
from constants import BIGGER_NEGATIVE_REWARD


class Score:

    """
    This class is used to store the scores, the mean scores, the max snake
    lengths and the number of turns of each game.
    It is also used to display/save the score evolution plot.
    """

    def __init__(self, args):

        """
        Constructor of the Score class.
        """

        self.score = 0              # Current score
        self.game_number = 0        # Current game number, +1 at each game
        self.turn = 0               # Current turn number, +1 at each turn
        self.snake_len = 0          # Current snake length
        self.max_snake_len = 0      # Max snake length of the current game

        self.scores = []
        self.iterations = []
        self.mean_scores = []
        self.max_snake_lengths = []
        self.nb_turns = []

        self.dont_save = args.dont_save

        if not args.new_model:
            self.load_scores(args.model_path)

        self.high_score = int(max(self.scores, default=0))

        self.training_sessions = 0
        if args.training_sessions:
            self.training_sessions = self.game_number + args.training_sessions

        plt.switch_backend("agg")

    def turn_update(self, reward, snake_len):

        """
        Use this method to update the score evolution at each turn.
        """

        # Update the score and the high score
        self.score += reward
        self.high_score = max(self.score, self.high_score)

        # Update the snake length and the max snake length
        self.snake_len = snake_len
        self.max_snake_len = max(self.snake_len, self.max_snake_len)

        # Update the turn
        self.turn += 1

    def game_over_update(self):

        """
        Use this method to update the score evolution at the end of a game.
        """

        self.scores.append(self.score - BIGGER_NEGATIVE_REWARD)
        self.mean_scores.append(int(sum(self.scores) / (self.game_number + 1)))
        self.iterations.append(self.game_number)
        self.max_snake_lengths.append(self.snake_len)
        self.nb_turns.append(self.turn)

        self.score = 0
        self.turn = 0
        self.max_snake_len = 0
        self.game_number += 1

    def save(self, directory):

        """
        Save the plot and the scores in the specified directory.
        2 files are created:
        - score_plot.png : the score evolution plot
        - score_data.csv : the score values dataframe
        """

        if self.dont_save:
            return

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.save_scores(os.path.join(directory, "score_data.csv"))
        self.save_plot(os.path.join(directory, "score_plot.png"))

    def save_plot(self, filename):

        """
        Save the score evolution plot as a .png file.
        If the display_plot attribute is set to False, the plot must be created
        before saving it.
        """

        fig, ax = plt.subplots()
        ax.plot(self.iterations, self.scores)
        ax.plot(self.iterations, self.mean_scores)

        ax.legend(["Score", "Mean Score"])
        ax.set_title("Learn2Slither score evolution")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Score")

        fig.savefig(filename)

        plt.close(fig)

        print(f"Score evolution plot saved as {filename}")

    def save_scores(self, filename):

        """
        Save the score values in a .csv file.
        """

        scores = pandas.DataFrame(
            {
                "Iterations": self.iterations,
                "Score": self.scores,
                "Mean Score": self.mean_scores,
                "Max Snake Lengths": self.max_snake_lengths,
                "Nb Turns": self.nb_turns,
            }
        )
        scores.to_csv(filename, sep="\t", index=False)
        print(f"Score values saved as {filename}")

    def load_scores(self, model_path):

        """
        Method used to load the scores from a file.

        If no model_path is specified, the scores are loaded from the last
        model directory. If no score file is found, the scores are not loaded.

        Else, the scores are loaded from the specified model directory.
        """

        if not model_path:

            # Load the scores from the last model directory
            model_path = os.path.join(os.getcwd(), "models")
            if os.path.exists(model_path):
                model_directory = os.listdir(model_path)
                if model_directory:
                    model_directory = max(
                        [int(directory.split("_")[1])
                         for directory in model_directory
                         if len(directory.split("_")) == 2]
                    )
                    model_path = os.path.join(
                        model_path, f"game_{model_directory}"
                    )
                    scores_filename = os.path.join(
                        model_path, "score_data.csv"
                    )
                    if os.path.exists(scores_filename):
                        self.load_scores_from_file(scores_filename)
        else:

            # Load the scores from the specified model directory
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Score directory {model_path} does not exist."
                )
            scores_filename = os.path.join(model_path, "score_data.csv")
            if not os.path.exists(scores_filename):
                raise FileNotFoundError(
                    f"Score evolution file {scores_filename} does not exist."
                )
            self.load_scores_from_file(scores_filename)

    def load_scores_from_file(self, filename: str):

        """
        Load the scores from a .csv file using pandas.
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Score evolution file {filename} does not exist."
            )

        csv = pandas.read_csv(filename, sep="\t")

        self.iterations = csv["Iterations"].tolist()
        self.scores = csv["Score"].tolist()
        self.mean_scores = csv["Mean Score"].tolist()
        self.max_snake_lengths = csv["Max Snake Lengths"].tolist()
        self.nb_turns = csv["Nb Turns"].tolist()
        self.game_number = len(self.iterations)

    def training_session_not_finished(self):
        return True if not self.training_sessions else (
            self.game_number < self.training_sessions
        )
