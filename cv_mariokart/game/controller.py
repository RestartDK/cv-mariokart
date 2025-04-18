import enum
import os
from typing import Optional, TextIO


@enum.unique
class Button(enum.Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11


@enum.unique
class Trigger(enum.Enum):
    L = 0
    R = 1


@enum.unique
class Stick(enum.Enum):
    MAIN = 0
    C = 1


class Controller:
    """Writes out controller inputs."""

    def __init__(self, path: str) -> None:
        """Create, but do not open the fifo."""
        self.pipe: Optional[TextIO] = None
        self.path: str = path
        try:
            os.mkfifo(self.path)
        except OSError:
            pass

    def __enter__(self) -> "Controller":
        """Opens the fifo. Blocks until the other side is listening."""
        self.pipe = open(self.path, "w", buffering=1)
        return self

    def __exit__(self, *args) -> None:
        """Closes the fifo."""
        if self.pipe:
            self.pipe.close()

    def press_button(self, button: Button) -> None:
        """Press a button."""
        assert button in Button
        if self.pipe:
            self.pipe.write("PRESS {}\n".format(button.name))

    def release_button(self, button: Button) -> None:
        """Release a button."""
        assert button in Button
        if self.pipe:
            self.pipe.write("RELEASE {}\n".format(button.name))

    def press_trigger(self, trigger: Trigger, amount: float) -> None:
        """Press a trigger. Amount is in [0, 1], with 0 as released."""
        assert trigger in Trigger
        assert 0 <= amount <= 1
        if self.pipe:
            self.pipe.write("SET {} {:.2f}\n".format(trigger.name, amount))

    def tilt_stick(self, stick: Stick, x: float, y: float) -> None:
        """Tilt a stick. x and y are in [0, 1], with 0.5 as neutral."""
        assert stick in Stick
        assert 0 <= x <= 1 and 0 <= y <= 1
        if self.pipe:
            self.pipe.write("SET {} {:.2f} {:.2f}\n".format(stick.name, x, y))

    def reset(self) -> None:
        """Reset all controls to their default state."""
        for button in Button:
            self.release_button(button)
        for trigger in Trigger:
            self.press_trigger(trigger, 0)
        for stick in Stick:
            self.tilt_stick(stick, 0.5, 0.5)
