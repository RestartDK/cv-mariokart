import os.path
import time
from cv_mariokart.controller import Button, Controller, Stick, Trigger


def find_dolphin_dir():
    """Attempts to find the dolphin user directory. None on failure."""
    candidates = [
        "~/.dolphin-emu",
        "~/.local/share/.dolphin-emu",
        "~/Library/Application Support/Dolphin/",
    ]
    for candidate in candidates:
        path = os.path.expanduser(candidate)
        if os.path.isdir(path):
            return path
    return None


def press_and_release(controller: Controller, button, duration=0.2):
    """Press a button, wait, then release it."""
    print(f"Pressing {button.name} for {duration} seconds")
    controller.press_button(button)
    time.sleep(duration)
    controller.release_button(button)
    time.sleep(0.1)  # Small pause after releasing


def drive_forward(controller: Controller, duration=2.0):
    """Drive forward by holding A button."""
    print("Driving forward...")
    controller.press_button(Button.A)
    time.sleep(duration)
    controller.release_button(Button.A)


def turn_left(controller: Controller, duration=1.0):
    """Turn left while driving."""
    print("Turning left...")
    controller.press_button(Button.A)  # Accelerate
    controller.tilt_stick(Stick.MAIN, 0.0, 0.5)  # Tilt stick left
    time.sleep(duration)
    controller.tilt_stick(Stick.MAIN, 0.5, 0.5)  # Center stick
    controller.release_button(Button.A)


def turn_right(controller: Controller, duration=1.0):
    """Turn right while driving."""
    print("Turning right...")
    controller.press_button(Button.A)  # Accelerate
    controller.tilt_stick(Stick.MAIN, 1.0, 0.5)  # Tilt stick right
    time.sleep(duration)
    controller.tilt_stick(Stick.MAIN, 0.5, 0.5)  # Center stick
    controller.release_button(Button.A)


def drift(controller: Controller, direction="left", duration=2.0):
    """Perform a drift (using B button + direction)."""
    print(f"Drifting {direction}...")
    controller.press_button(Button.A)  # Accelerate
    time.sleep(0.5)

    # Tilt stick for direction
    if direction == "left":
        controller.tilt_stick(Stick.MAIN, 0.0, 0.5)  # Full left
    else:
        controller.tilt_stick(Stick.MAIN, 1.0, 0.5)  # Full right

    controller.press_trigger(Trigger.R, 1.0)
    time.sleep(duration)

    # Release all
    controller.press_trigger(Trigger.R, 0.5)
    controller.tilt_stick(Stick.MAIN, 0.5, 0.5)  # Center stick
    controller.release_button(Button.A)


def use_item(controller: Controller):
    print("Using item...")
    controller.press_trigger(Trigger.L, 1.0)
    controller.press_trigger(Trigger.L, 0.5)


def drive_in_circle(controller: Controller, clockwise=True, cycles=2):
    """Drive in circles."""
    print(f"Driving in {'clockwise' if clockwise else 'counter-clockwise'} circles...")
    stick_x = 0.75 if clockwise else 0.25  # Less than full tilt for a controlled circle

    controller.press_button(Button.A)  # Start accelerating

    for _ in range(cycles):
        controller.tilt_stick(Stick.MAIN, stick_x, 0.5)
        time.sleep(3.0)  # Full circle takes ~3 seconds

    controller.tilt_stick(Stick.MAIN, 0.5, 0.5)  # Center stick
    controller.release_button(Button.A)


def zigzag(controller: Controller, cycles=3, duration=0.5):
    """Drive in a zigzag pattern."""
    print("Driving in zigzag pattern...")
    controller.press_button(Button.A)

    for _ in range(cycles):
        controller.tilt_stick(Stick.MAIN, 0.2, 0.5)  # Left
        time.sleep(duration)

        controller.tilt_stick(Stick.MAIN, 0.8, 0.5)  # Right
        time.sleep(duration)

    controller.tilt_stick(Stick.MAIN, 0.5, 0.5)  # Center stick
    controller.release_button(Button.A)


def race_start(controller: Controller):
    """Perform a rocket start at the beginning of the race."""
    print("Preparing for rocket start...")
    # Press A when the 2 appears
    press_and_release(controller, Button.A, 0.1)
    time.sleep(1.5)  # Wait for countdown

    # Press and hold A right before the race starts
    controller.press_button(Button.A)
    time.sleep(2.0)
    # Continue driving...


def demo_triggers(controller: Controller):
    """Demonstrate using triggers (for games that use them)."""
    print("Testing triggers...")

    # Gradually press R trigger
    for i in range(11):
        value = i / 10.0
        print(f"Setting R trigger to {value:.1f}")
        controller.press_trigger(Trigger.R, value)
        time.sleep(0.2)

    # Release trigger
    controller.press_trigger(Trigger.R, 0.0)
    time.sleep(0.5)


def run(controller: Controller):
    """Run a series of test movements for Mario Kart."""
    # Allow time to get to the starting line
    print("Starting tests in 5 seconds...")
    time.sleep(5)

    # Start the race with a rocket boost
    race_start(controller)

    # Basic movements
    drive_forward(controller, 3.0)
    time.sleep(0.5)

    turn_left(controller, 1.0)
    time.sleep(0.5)

    turn_right(controller, 1.0)
    time.sleep(0.5)

    # Advanced techniques
    drift(controller, "left", 2.0)
    time.sleep(1.0)

    drift(controller, "right", 2.0)
    time.sleep(1.0)

    # Use an item
    use_item(controller)
    time.sleep(1.0)

    # Complex patterns
    drive_in_circle(controller, clockwise=True, cycles=1)
    time.sleep(1.0)

    drive_in_circle(controller, clockwise=False, cycles=1)
    time.sleep(1.0)

    zigzag(controller, cycles=3)
    time.sleep(1.0)

    # Demonstrate triggers (if the game uses them)
    # demo_triggers(controller)

    # Final sprint
    print("Final sprint to the finish!")
    controller.press_button(Button.A)
    time.sleep(5.0)
    controller.release_button(Button.A)

    # Reset all controls
    controller.reset()

    print("Test complete!")


def main():
    dolphin_dir = find_dolphin_dir()
    if dolphin_dir is None:
        print("Could not find dolphin config dir.")
        return

    try:
        print("Start dolphin now. Press ^C to stop.")
        ctrl_path = dolphin_dir + "/Pipes/pipe"
        with Controller(ctrl_path) as ctrl:
            run(ctrl)
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
