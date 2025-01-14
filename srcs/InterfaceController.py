class InterfaceController:

    def __init__(
                self,
                is_ai: bool = False,
                cli_enabled: bool = True,
                gui_enabled: bool = True
            ):

        self.ai_enabled = is_ai
        self.cli_enabled = cli_enabled
        self.gui_enabled = gui_enabled

    def toggle_ai(self):
        self.ai_enabled = not self.ai_enabled
        if self.is_ai():
            print("AI mode enabled")
        else:
            print("Human mode enabled")

    def toggle_cli(self):
        self.cli_enabled = not self.cli_enabled
        if self.cli_enabled:
            print("CLI enabled")
        else:
            print("\033c")
            print("CLI disabled, the game will run in the background")
            print("Press 'c' to enable the CLI")

    def toggle_gui(self, gui, environment, score_evolution):
        self.gui_enabled = not self.gui_enabled
        if self.gui_enabled:
            print("GUI enabled")
            gui.draw(environment, score_evolution, self)
        else:
            print("GUI disabled")
            gui.disable()

    def is_ai(self):
        return self.ai_enabled

    def is_human(self):
        return not self.ai_enabled

    def cli_disabled(self):
        return not self.cli_enabled

    def gui_disabled(self):
        return not self.gui_enabled
