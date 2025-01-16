import matplotlib.pyplot as plt
import pandas
import os
from constants import BIGGER_NEGATIVE_REWARD


class ScoreEvolutionPlot:

    """
    This class is used to store the scores, the mean scores, the max snake
    lengths and the number of turns of each game.
    It is also used to display/save the score evolution plot.
    """

    def __init__(self, args):

        """
        Constructor of the ScoreEvolutionPlot class.
        """

        self.score = 0
        self.game_number = 0
        self.turn = 0
        self.snake_len = 0
        self.max_snake_len = 0

        self.scores = []
        self.iterations = []
        self.mean_scores = []
        self.max_snake_lengths = []
        self.nb_turns = []

        self.display_plot = args.plot
        self.dont_save = args.dont_save

        if args.new_model is False:
            self.load_scores(args.model_path)

        self.high_score = int(max(self.scores, default=0))

        self.training_sessions = 0
        if args.training_sessions:
            self.training_sessions = self.game_number + args.training_sessions

        self.init_plot()

    def __del__(self):

        """
        Destructor of the ScoreEvolutionPlot class.
        Disables the plot display and closes the plot window.
        """

        if self.display_plot:
            plt.close()
            plt.ioff()

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
        self.mean_scores.append(int(sum(self.scores) / len(self.scores)))
        self.iterations.append(self.game_number)
        self.max_snake_lengths.append(self.snake_len)
        self.nb_turns.append(self.turn)

        self.score = 0
        self.turn = 0
        self.game_number += 1

        if not self.display_plot:
            return

        self.set_plot_values()

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

        self.save_plot(os.path.join(directory, "score_plot.png"))
        self.save_scores(os.path.join(directory, "score_data.csv"))

    def save_plot(self, filename):

        """
        Save the score evolution plot as a .png file.
        If the display_plot attribute is set to False, the plot must be created
        before saving it.
        """

        if not self.display_plot:
            self.display_plot = True
            self.init_plot()

        self.figure.savefig(filename)
        print(f"Score evolution plot saved as {filename}")

    def save_scores(self, filename):

        """
        Save the score values in a .csv file.
        """

        scores = pandas.DataFrame(
            {
                "Iterations": self.iterations,
                "Scores": self.scores,
                "Mean Scores": self.mean_scores,
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
                    model_directory = sorted(
                        [int(directory.split("_")[1])
                         for directory in model_directory]
                    )
                    model_path = os.path.join(
                        model_path,
                        f"game_{model_directory[-1]}"
                    )
                    scores_filename = os.path.join(
                        model_path,
                        "score_evolution.csv"
                    )
                    if os.path.exists(scores_filename):
                        self.load_scores_from_file(scores_filename)
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Score directory {model_path} does not exist.")
            scores_filename = os.path.join(model_path, "score_evolution.csv")
            if not os.path.exists(scores_filename):
                raise FileNotFoundError(f"Score evolution file {scores_filename} does not exist.")
            self.load_scores_from_file(scores_filename)

    def load_scores_from_file(self, filename):
        if not os.path.exists(filename):
            return

        scores = pandas.read_csv(filename, sep="\t")

        self.iterations = scores["Iterations"].tolist()
        self.scores = scores["Scores"].tolist()
        self.mean_scores = scores["Mean Scores"].tolist()
        self.max_snake_lengths = scores["Max Snake Lengths"].tolist()
        self.nb_turns = scores["Nb Turns"].tolist()
        self.game_number = len(self.iterations)

    def training_session_not_finished(self):
        return True if not self.training_sessions else (
            self.game_number < self.training_sessions
        )

    def init_plot(self):

        """
        Initialize the score evolution plot.
        """

        if not self.display_plot:
            return

        # Set matplotlib to interactive mode
        plt.ion()

        # Create the plot
        self.figure, self.ax = plt.subplots()
        self.score_plot, = self.ax.plot(self.iterations, self.scores)
        self.mean_score_plot, = self.ax.plot(self.iterations, self.mean_scores)

        # Legend, titles and labels
        self.ax.legend(["Score", "Mean Score"])
        self.ax.set_title("Learn2Slither score evolution")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Score")

    def set_plot_values(self):

        """
        Set the plot values.
        """

        self.score_plot.set_xdata(self.iterations)
        self.score_plot.set_ydata(self.scores)

        self.mean_score_plot.set_xdata(self.iterations)
        self.mean_score_plot.set_ydata(self.mean_scores)

        self.ax.relim()
        self.ax.autoscale_view()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

