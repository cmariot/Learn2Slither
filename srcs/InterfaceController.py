class InterfaceController:

    def __init__(
                self,
                args: tuple,
                is_ai: bool = True,
                cli_enabled: bool = True,
                gui_enabled: bool = True
            ):

        self.step_by_step = args.step_by_step
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

    def toggle_gui(self, gui, environment, scores):
        self.gui_enabled = not self.gui_enabled
        if self.gui_enabled:
            print("GUI enabled")
            gui.game.draw(environment, scores, self)
        else:
            print("GUI disabled")
            gui.game.disable(gui)

    def is_ai(self):
        return self.ai_enabled

    def is_human(self):
        return not self.ai_enabled

    def cli_disabled(self):
        return not self.cli_enabled

    def gui_disabled(self):
        return not self.gui_enabled

    def change_fps(self, key, gui, cli, shift_pressed):

        # Increase or decrease the FPS of the game based on the key pressed
        # Keys : '[+]' to increase the FPS and '[-]' to decrease the FPS

        if key == '=' and shift_pressed:
            key = '[+]'
        elif key == '-':
            key = '[-]'

        if key not in ['[+]', '[-]']:
            return

        change = -10 if key == '[-]' else 10
        if gui.fps <= 10 and change == -10:
            change = -1
        elif gui.fps <= 9 and change == 10:
            change = 1

        fps = gui.fps + change

        if fps <= 0:
            print("FPS must be greater than 0")
            return

        gui.set_fps(fps)
        cli.set_fps(fps)
        print(f"FPS: {gui.fps}")

    def toggle_step_by_step(self):
        self.step_by_step = not self.step_by_step
        if self.step_by_step:
            print("Step by step mode enabled")
        else:
            print("Step by step mode disabled")
