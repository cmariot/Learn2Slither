class GameMode:

    HUMAN = 1
    AI = 2

    def __init__(self, mode: str):
        if mode == "human":
            self.mode = self.HUMAN
        elif mode == "ai":
            self.mode = self.AI
        else:
            raise ValueError("Invalid game mode")

    def switch(self):
        self.mode = self.HUMAN if self.mode == self.AI else self.AI
        print(f"Switching to {self}")

    def is_human(self):
        return self.mode == self.HUMAN

    def is_ai(self):
        return self.mode == self.AI

    def __str__(self):
        return "human" if self.mode == self.HUMAN else "ai"
