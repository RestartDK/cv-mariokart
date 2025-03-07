import os.path
from .controller import Controller


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


def main():
    dolphin_dir = find_dolphin_dir()
    if dolphin_dir is None:
        print("Could not find dolphin config dir.")
        return

    ctrl = None
    try:
        print("Start dolphin now. Press ^C to stop.")
        ctrl_path = dolphin_dir + "/Pipes/pipe"
        ctrl = Controller(ctrl_path)
        # run(ctrl)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if ctrl is not None:
            print("Resetting controller")
            ctrl.reset()
            print("Controller reset complete")
        print("Stopped")


if __name__ == "__main__":
    main()
