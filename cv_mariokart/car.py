from .controller import Controller, Trigger, Button, Stick


class Car:
    def __init__(self, ctrl: Controller) -> None:
        self.ctrl = ctrl

    def drive_forward(self):
        """Drive forward by holding A button."""
        print("Driving forward...")
        self.ctrl.press_button(Button.A)

    def stop_car(self):
        """Release A to stop the car"""
        print("Stopping car...")  # Fixed the message here
        self.ctrl.release_button(Button.A)

    def turn(self, x: float):
        """Turn right with x > 0.5 and turn left when x < 0.5"""
        self.ctrl.tilt_stick(Stick.MAIN, x, 0.5)

    def drift(self):
        """Perform a drift (using R Trigger + direction)."""
        print("Drifting...")
        self.ctrl.press_trigger(Trigger.R, 1.0)

    def stop_drift(self):
        """Perform a drift (using R Trigger + direction)."""
        print("Stopping drift...")  # Fixed the message here
        self.ctrl.press_trigger(Trigger.R, 0.0)

    def use_item(self):
        """Use item with the L Trigger amount does not matter"""
        print("Using item...")
        self.ctrl.press_trigger(Trigger.L, 1.0)
        self.ctrl.press_trigger(Trigger.L, 0.5)
