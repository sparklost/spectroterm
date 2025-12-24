import shutil
import signal
import sys

# for reading pressed key
# import termios
# import tty
# STDIN_FD = sys.stdin.fileno()
# old_term = termios.tcgetattr(STDIN_FD)

resized = True
width = 0
height = 0

def handle_resize(signum, frame):   # noqa
    """Handle terminal resize event"""
    global resized
    resized = True


signal.signal(signal.SIGWINCH, handle_resize)


def enter_raw():
    """Enter raw terminal mode"""
    # tty.setraw(STDIN_FD)
    sys.stdout.write(
        "\x1b[?1049h"   # alternate screen
        "\x1b[?25l"     # hide cursor
        "\x1b[2J"       # clear screen
        "\x1b[H"        # cursor home
        "\x1b[?7l",     # disable line wrap
    )
    sys.stdout.flush()


def leave_raw():
    """Leave raw terminal mode"""
    sys.stdout.write(
        "\x1b[?25h"     # show cursor
        "\x1b[?7h"      # enable line wrap
        "\x1b[?1049l"   # leave alternate screen
        "\x1b[0m",      # reset attrs
    )
    sys.stdout.flush()
    # termios.tcsetattr(STDIN_FD, termios.TCSADRAIN, old_term)


def get_size():
    """Get size of terminal in characters (h, w)"""
    size = shutil.get_terminal_size()
    return size.lines, size.columns


# def read_key():
#     """Read pressed key"""
#     r, _, _ = select.select([sys.stdin], [], [], 0)
#     if r:
#         return sys.stdin.read(1)
#     return None


def draw(lines):
    """Draw lines on screen"""
    sys.stdout.write("\x1b[H")   # cursor home
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()
