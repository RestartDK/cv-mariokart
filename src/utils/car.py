import time

from controller import Button, Controller, Stick, Trigger


class Car:
    def __init__(self, ctrl: Controller) -> None:
        self.ctrl = ctrl

    def drive_forward(self):
        """Drive forward by holding A button."""
        print("Driving forward...")
        self.ctrl.press_button(Button.A)

    def stop_car(self):
        """Release A to stop the car."""
        print("Stopping car...")
        self.ctrl.release_button(Button.A)

    def turn(self, x: float):
        """Turn right when x > 0.5 and turn left when x < 0.5."""
        self.ctrl.tilt_stick(Stick.MAIN, x, 0.5)

    def drift(self):
        """Perform a drift using the R Trigger."""
        print("Drifting...")
        self.ctrl.press_trigger(Trigger.R, 1.0)

    def stop_drift(self):
        """Stop drifting by releasing the R Trigger."""
        print("Stopping drift...")
        self.ctrl.press_trigger(Trigger.R, 0.0)

    def use_item(self):
        """Use an item by activating the L Trigger."""
        print("Using item...")
        self.ctrl.press_trigger(Trigger.L, 1.0)
        time.sleep(1)
        self.ctrl.press_trigger(Trigger.L, 0.0)
