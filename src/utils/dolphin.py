import os


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
