import matplotlib.pyplot as plt
import pandas
import os
from constants import BIGGER_NEGATIVE_REWARD


class ScoreEvolutionPlot:

    is_macos = os.name == "posix" and os.uname().sysname == "Darwin"
    is_macos = True

    def __init__(self, model_path=None, training_sessions=0):

        self.game_number = 0
        self.turn = 0
        self.score = 0

        self.scores = []
        self.iterations = []
        self.mean_scores = []

        self.load_scores(model_path)

        self.high_score = int(max(self.scores, default=0))

        # The number of training sessions that the agent will perform
        if training_sessions > 0:
            self.training_sessions = training_sessions + self.game_number
        else:
            self.training_sessions = 0

        if not self.is_macos:

            plt.ion()

            self.figure, self.ax = plt.subplots()

            self.score_plot, = self.ax.plot(
                self.scores, self.iterations
            )
            self.mean_score_plot, = self.ax.plot(
                self.mean_scores, self.iterations
            )

            # Legend, titles and labels
            self.ax.legend(["Score", "Mean Score"])
            self.ax.set_title("Learn2Slither score evolution")
            self.ax.set_xlabel("Iterations")
            self.ax.set_ylabel("Score")

    def __del__(self):
        if not self.is_macos:
            plt.ioff()

    def update(self):

        self.game_number += 1
        self.turn = 0

        self.score -= BIGGER_NEGATIVE_REWARD

        self.scores.append(self.score)
        self.mean_scores.append(int(sum(self.scores) / len(self.scores)))
        self.iterations.append(self.game_number)

        if self.score > self.high_score:
            self.high_score = self.score

        self.score = 0

        if self.is_macos:
            return

        self.score_plot.set_xdata(self.iterations)
        self.score_plot.set_ydata(self.scores)

        self.mean_score_plot.set_xdata(self.iterations)
        self.mean_score_plot.set_ydata(self.mean_scores)

        self.ax.relim()
        self.ax.autoscale_view()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save(self, directory):
        plot_filename = os.path.join(directory, "score_evolution.png")
        scores_filename = os.path.join(directory, "score_evolution.csv")
        self.save_plot(plot_filename)
        self.save_scores(scores_filename)

    def save_plot(self, filename):
        if not self.is_macos:
            self.figure.savefig(filename)
        else:
            plt.figure()
            plt.plot(self.iterations, self.scores)
            plt.plot(self.iterations, self.mean_scores)
            plt.legend(["Score", "Mean Score"])
            plt.title("Learn2Slither score evolution")
            plt.xlabel("Iterations")
            plt.ylabel("Score")
            plt.savefig(filename)
        print(f"Score evolution plot saved as {filename}")

    def save_scores(self, filename):
        scores = pandas.DataFrame(
            [
                self.iterations,
                self.scores,
                self.mean_scores
            ],
            index=["Iterations", "Scores", "Mean Scores"]
        ).T
        scores.to_csv(filename, sep="\t", index=False)
        print(f"Score values saved as {filename}")

    def load_scores(self, model_path):
        if not model_path:
            model_path = os.path.join(os.getcwd(), "models")
            if os.path.exists(model_path):
                model_directory = os.listdir(model_path)
                if model_directory:
                    for i in range(len(model_directory)):
                        model_directory[i] = int(
                            model_directory[i].split("_")[1]
                        )
                    model_directory.sort()
                    model_directory = f"game_{model_directory[-1]}"
                    model_path = os.path.join(model_path, model_directory)
                    scores_filename = os.path.join(
                        model_path, "score_evolution.csv"
                    )
                    if os.path.exists(scores_filename):
                        self.load_scores_from_file(scores_filename)
        else:
            scores_filename = os.path.join(model_path, "score_evolution.csv")
            if os.path.exists(scores_filename):
                self.load_scores_from_file(scores_filename)

    def load_scores_from_file(self, filename):
        if not os.path.exists(filename):
            return

        scores = pandas.read_csv(filename, sep="\t")

        self.iterations = scores["Iterations"].tolist()
        self.scores = scores["Scores"].tolist()
        self.mean_scores = scores["Mean Scores"].tolist()
        self.game_number = int(self.iterations[-1])

    def training_session_not_finished(self):
        if self.training_sessions == 0:
            return True

        print(
            f"Training session {self.game_number}/{self.training_sessions}",
            f"{self.game_number < self.training_sessions}"
        )
        return self.game_number < self.training_sessions
