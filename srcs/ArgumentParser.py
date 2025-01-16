import argparse


class IsPositiveCondition(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values <= 0:
            raise argparse.ArgumentTypeError(
                f"Parsing error: {self.dest} must be positive and greater than 0"
            )
        setattr(namespace, self.dest, values)


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="Learn2Slither",
            description='Learn2Slither is a reinforcement learning snake game.',
            epilog='Enjoy the game!'
        )

        # Argument to define how many training sessions should be executed
        self.parser.add_argument(
            "--training-sessions",
            type=int,
            default=0,
            help="Number of training sessions to execute.",
            action=IsPositiveCondition
        )

        # Argument to define the FPS of the game
        self.parser.add_argument(
            "--fps",
            type=int,
            default=20,
            help="Frames per second of the game.",
            action=IsPositiveCondition
        )

        # AI in step by step mode
        self.parser.add_argument(
            "--step-by-step",
            action="store_true",
            help="Step by step mode for the AI. Press 'enter' to continue."
        )

        # Path of the model directory
        self.parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="Path of the model directory."
        )

        # Boolean disable model and score saving
        self.parser.add_argument(
            "--dont-save",
            action="store_true",
            help="Do not save the model."
        )

        # Boolean to train a model from scratch
        self.parser.add_argument(
            "--new-model",
            action="store_true",
            help="Train a model from scratch."
        )

        # Parse the arguments
        self.args = self.parse_args()

        # Incompatible arguments :
        # - --model-path and --new-model
        if self.args.model_path is not None and self.args.new_model:
            self.parser.error(
                "--model-path and --new-model are incompatible arguments."
            )

    def parse_args(self):
        return self.parser.parse_args()
