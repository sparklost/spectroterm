import argparse
import curses
import importlib.util
import shutil
import signal
import subprocess
import sys
import time
from collections import deque

import numpy as np
import soundcard as sc
from pyfftw.interfaces.numpy_fft import rfft


def log_band_volumes(data, freqs, num_bands, band_edges, max_ref):
    """Get logarythmic volume in dB for specified number of bands, from sound sample, with interpolation between bands"""
    # get magnitude from fft
    raw_magnitude = np.abs(rfft(data, threads=1))

    # split into logarithmic bands
    magnitude = np.zeros(num_bands)
    for i in range(num_bands):
        left = band_edges[i]
        right = band_edges[i+1]
        idx = np.where((freqs >= left) & (freqs < right))[0]

        # interpolate between bands
        if len(idx) == 0:
            left_bin = np.searchsorted(freqs, left)
            right_bin = np.searchsorted(freqs, right)
            bins = []
            if left_bin > 0:
                bins.append(left_bin - 1)
            if left_bin < len(freqs):
                bins.append(left_bin)
            if right_bin > 0 and right_bin != left_bin:
                bins.append(right_bin - 1)
            if right_bin < len(freqs) and right_bin != left_bin:
                bins.append(right_bin)
            bins = np.array(list(set(bins)))
            weights = []
            for b in bins:
                center = freqs[b]
                band_center = (left + right) / 2
                d = abs(center - band_center) + 1e-6
                weights.append(1/d)
            weights = np.array(weights)
            weights /= np.sum(weights)
            magnitude[i] = np.sqrt(np.sum((raw_magnitude[bins]**2) * weights))   # weighted RMS
        else:
            magnitude[i] = np.sqrt(np.mean(raw_magnitude[idx]**2))   # RMS

    # magnitude to negative dB
    db = 20 * np.log10(magnitude / max_ref + 1e-12)    # add small value to avoid log(0)
    return np.maximum(db, -90)


def get_color(y, bar_height, use_color):
    """Get color id by bar height"""
    if not use_color:
        return curses.color_pair(0)
    relative = (bar_height - y) / bar_height
    if relative < 0.5:
        return curses.color_pair(1)   # green
    if relative < 0.8:
        return curses.color_pair(2)   # yellow
    return curses.color_pair(3)   # red


def draw_spectrum(spectrum_win, bar_heights, peak_heights, bar_height, bar_character, peak_character, peaks, color, box):
    """Draw spectrum bars with peaks"""
    width = bar_heights.shape[0]
    for y in range(bar_height - box):
        line = [" "] * width
        for i in range(width):
            bar = bar_heights[i]
            if y >= bar_height - bar:
                line[i] = bar_character
            if peaks and y == bar_height - peak_heights[i]:
                line[i] = peak_character
        spectrum_win.insstr(y, 0, "".join(line), get_color(y, bar_height, color))
        spectrum_win.refresh()


# use cython if available
if importlib.util.find_spec("spectrum_curses_cython"):
    from spectrum_cython import draw_spectrum, log_band_volumes


pw_loopback = None


def connect_pipewire(output_node_name, target_node_name=None, only_get_name=False):
    """Connect to output with custom loopback device. This prevents headsets from switching to 'handsfree' mode"""
    global pw_loopback
    # check if pipewire is running
    if "pipewire" not in subprocess.check_output(["ps", "-A"], text=True):
        sys.exit("Pipewire process not found")

    # check if pipewire commands are available
    if not (shutil.which("pw-link") or shutil.which("pw-loopback")):
        sys.exit("pw-link and pw-loopback commands not found")

    if target_node_name and ":" in target_node_name:
        target_node_name = target_node_name.split(":")[0]

    # find node that output is connected to
    command = ["pw-link", "--links"]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    links = proc.communicate()[0].decode().split("\n")
    last_nodes = []
    for num, link in enumerate(links):
        if target_node_name and target_node_name in link:
            last_nodes.append(target_node_name)
            break
        if not target_node_name and f"|-> {output_node_name}" in link:
            node_name = links[num-1].split(":")[0].strip()
            if node_name not in last_nodes:
                last_nodes.append(node_name)
    if not last_nodes:
        sys.exit("Could not find active pipewire links. Make sure audio is playing when starting spectroterm or specify custom node name.")

    if only_get_name:
        return last_nodes

    # start loopback node
    command = [
        "pw-loopback",
        "--capture-props", 'node.autoconnect=false node.name=spectroterm-capture node.description="Spectroterm Capture"',
        "--playback-props", 'node.autoconnect=false media.class=Audio/Source node.name=spectroterm node.description="Spectroterm"',
    ]
    pw_loopback = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.1)   # delay for pw-loopback to create nodes

    # link loopback node
    for node in last_nodes:
        proc = subprocess.Popen(
            ["pw-link", node, "spectroterm-capture"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return "spectroterm"


def db_to_height(db, min_db, max_db, bar_height):
    """Calculate height of bars from sound volume"""
    return np.clip(np.round(np.interp(db, (min_db, max_db), (0, bar_height))).astype(np.int32), 0, bar_height)


def draw_log_x_axis(screen, num_bars, x, h, min_freq, max_freq, have_box=True):
    """Draw logarythmic Hz x axis"""
    freqs = [30, 100, 200, 500, 1000, 2000, 5000, 10000, 16000]
    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bars + 1)
    for freq in freqs:
        if band_edges[0] < freq < band_edges[-1]:
            pos = np.argmin(np.abs(band_edges - freq))
            if 0 <= pos < num_bars:
                if freq >= 1000:
                    label = f"{round(freq/1000)}k"
                else:
                    label = str(round(freq))
                if pos < num_bars - 5:
                    screen.addstr(h - 1 - have_box, x + pos, label)
        screen.addstr(h - 1 - have_box, x + num_bars - 3 + have_box * 2, "Hz")


def draw_log_y_axis(screen, bar_height, min_db, max_db, have_box=True):
    """Draw logarythmic dB y axis"""
    levels = list(range(int(min_db), int(max_db) + 1, 10))
    for db in levels:
        # get y coordinate
        pos = int(np.interp(db, (min_db, max_db), (bar_height, 0)))
        label = str(db).rjust(3)
        if 0 <= pos < bar_height:
            screen.addstr(have_box + pos, have_box, label)
    screen.addstr(have_box, have_box + 1, "dB")


def draw_ui(screen, draw_box, draw_axes, min_freq, max_freq, min_db, max_db):
    """Draw UI"""
    h, w = screen.getmaxyx()
    screen.clear()
    spectrum_hwyx = (
        h - draw_box - draw_axes,
        w - 2 * draw_box - 4 * draw_axes,
        draw_box,
        draw_box + 4 * draw_axes,
    )
    spectrum_win = screen.derwin(*spectrum_hwyx)
    bar_height, num_bars = spectrum_win.getmaxyx()
    if draw_box:
        screen.box()
        screen.addstr(0, 2, "Spectrum Analyzer")
    if draw_axes:
        draw_log_y_axis(screen, bar_height, min_db, max_db, draw_box)
        draw_log_x_axis(screen, num_bars, 4, h, min_freq, max_freq, draw_box)
    return spectrum_win


def main(screen, args):
    """Main app function"""
    curses.curs_set(0)
    screen.nodelay(True)
    curses.start_color()
    curses.use_default_colors()

    # prevent mouse icon changing when running in tmux
    curses.mousemask(curses.ALL_MOUSE_EVENTS)
    curses.mouseinterval(0)

    # load config
    color = args.color
    box = args.box
    axes = args.axes
    peaks = args.peaks
    fall_speed = args.fall_speed
    bar_character = args.bar_character[0]
    peak_character = args.peak_character[0]
    sample_rate = args.sample_rate
    sample_size = args.sample_size / 1000
    reference_max = args.reference_max
    peak_hold = args.peak_hold / 1000
    min_freq = args.min_freq
    max_freq = args.max_freq
    min_db = args.min_db
    max_db = args.max_db
    pipewire_fix = args.pipewire_fix
    pipewire_node_id = args.pipewire_node_id
    delay = args.delay

    curses.init_pair(0, -1, -1)
    if color:
        curses.init_pair(1, args.green, -1)
        curses.init_pair(2, args.orange, -1)
        curses.init_pair(3, args.red, -1)

    # detect bluetooth device
    if args.bt_delay:
        if "blue" in sc.default_speaker().id:
            delay = args.bt_delay

    prev_bar_heights = None
    prev_update_time = time.perf_counter()
    peak_heights = np.array([], dtype="int32")
    numframes = int(sample_rate * sample_size)
    delay_frames = int(sample_rate * delay / 1000)
    freqs = np.fft.rfftfreq(numframes, 1 / sample_rate)

    # get loopback device
    if pipewire_fix:
        mic_id = connect_pipewire(sc.default_speaker().id, pipewire_node_id)
        if not mic_id:
            mic_id = sc.default_speaker().name
    else:
        mic_id = sc.default_speaker().name
    loopback_mic = sc.get_microphone(mic_id, include_loopback=True)

    try:
        with loopback_mic.recorder(samplerate=sample_rate, channels=1, blocksize=numframes) as rec:
            h, w = screen.getmaxyx()
            spectrum_win = draw_ui(screen, box, axes, min_freq, max_freq, min_db, max_db)
            bar_height, num_bars = spectrum_win.getmaxyx()
            band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bars + 1)
            silence = np.repeat(-90.0, num_bars)
            silence_time = 0
            max_silence_time = max(peak_hold, h/fall_speed) * 2 * 1000

            if delay:
                buffer = deque()
                buffer.extend(np.array_split(np.zeros((delay_frames)), delay_frames // numframes))

            while True:
                # handle input
                key = screen.getch()
                if key == 113:
                    break
                elif key == curses.KEY_RESIZE:
                    h, w = screen.getmaxyx()
                    spectrum_win = draw_ui(screen, box, axes, min_freq, max_freq, min_db, max_db)
                    bar_height, num_bars = spectrum_win.getmaxyx()
                    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bars + 1)
                    silence = np.repeat(-90, num_bars)

                # get and process data
                if delay:
                    buffer.append(rec.record(numframes=numframes).flatten())
                    data = buffer.popleft()
                else:
                    data = rec.record(numframes=numframes).flatten()
                # skip calculations if all data is zero
                if data.any():
                    db = log_band_volumes(data, freqs, num_bars, band_edges, reference_max)
                    silence_time = 0
                else:
                    if silence_time >= max_silence_time:
                        continue
                    silence_time += int(args.sample_size)
                    db = silence
                # calculate heights on screen
                raw_bar_heights = db_to_height(db, min_db, max_db, bar_height)

                # falling bars
                now = time.perf_counter()
                dt = now - prev_update_time
                prev_update_time = now
                if prev_bar_heights is None or len(prev_bar_heights) != len(raw_bar_heights):
                    prev_bar_heights = raw_bar_heights.copy()
                else:
                    max_fall = int(fall_speed * dt)
                    for i in range(len(raw_bar_heights)):
                        if raw_bar_heights[i] >= prev_bar_heights[i]:
                            prev_bar_heights[i] = raw_bar_heights[i]
                        else:
                            prev_bar_heights[i] = max(raw_bar_heights[i], prev_bar_heights[i] - max_fall)
                bar_heights = prev_bar_heights

                # peak marker
                if peaks:
                    if len(peak_heights) != len(bar_heights):
                        peak_heights = bar_heights.copy()
                        peak_times = [now] * len(bar_heights)
                    for i, bh in enumerate(bar_heights):
                        if bh > peak_heights[i]:
                            peak_heights[i] = bh
                            peak_times[i] = now
                        elif now - peak_times[i] > peak_hold:
                            peak_heights[i] = bh
                            peak_times[i] = now

                # draw spectrum
                try:
                    draw_spectrum(spectrum_win, bar_heights, peak_heights, bar_height, bar_character, peak_character, peaks, color, box)
                except curses.error:
                    h, w = screen.getmaxyx()
                    spectrum_win = draw_ui(screen, box, axes, min_freq, max_freq, min_db, max_db)
                    bar_height, num_bars = spectrum_win.getmaxyx()
                    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bars + 1)
                    silence = np.repeat(-90, num_bars)

    except Exception as e:
        if pw_loopback:
            pw_loopback.send_signal(signal.SIGINT)
            pw_loopback.wait()
        sys.exit(f"Error: {e}")


def sigint_handler(signum, frame):   # noqa
    """Handle Ctrl-C event"""
    if pw_loopback:
        pw_loopback.send_signal(signal.SIGINT)
        pw_loopback.wait()
    sys.exit(0)


def argparser():
    """Setup argument parser for CLI"""
    parser = argparse.ArgumentParser(
        prog="spectroterm",
        description="Curses based terminal spectrum analyzer for currently playing audio",
    )
    parser._positionals.title = "arguments"
    parser.add_argument(
        "-a",
        "--axes",
        action="store_true",
        help="draw graph axes",
    )
    parser.add_argument(
        "-b",
        "--box",
        action="store_true",
        help="draw lines at terminal borders",
    )
    parser.add_argument(
        "-c",
        "--color",
        action="store_true",
        help="3 color mode",
    )
    parser.add_argument(
        "-p",
        "--peaks",
        action="store_true",
        help="draw peaks that disappear after some time",
    )
    parser.add_argument(
        "-f",
        "--fall-speed",
        type=int,
        default=40,
        help="speed at which bars fall in characters per second",
    )
    parser.add_argument(
        "-o",
        "--peak-hold",
        type=int,
        default=2000,
        help="time after which peak will dissapear, in ms",
    )
    parser.add_argument(
        "-r",
        "--bar-character",
        type=str,
        default="█",
        help="character used to draw bars",
    )
    parser.add_argument(
        "-k",
        "--peak-character",
        type=str,
        default="_",
        help="character used to draw peaks",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=30,
        help="minimum frequency on spectrum graph (x-axis)",
    )
    parser.add_argument(
        "--max-freq",
        type=int,
        default=16000,
        help="maximum frequency on spectrum graph (x-axis)",
    )
    parser.add_argument(
        "--min-db",
        type=int,
        default=-90,
        help="minimum loudness on spectrum graph (y-axis)",
    )
    parser.add_argument(
        "--max-db",
        type=int,
        default=0,
        help="maximum loudness on spectrum graph (y-axis)",
    )
    parser.add_argument(
        "--green",
        type=int,
        default=46,
        help="8bit ANSI color code for green part of bar",
    )
    parser.add_argument(
        "--orange",
        type=int,
        default=214,
        help="8bit ANSI color code for orange part of bar",
    )
    parser.add_argument(
        "--red",
        type=int,
        default=196,
        help="8bit ANSI color code for red part of bar",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="spectrogram delay for a better sync with sound.",
    )
    parser.add_argument(
        "--bt-delay",
        type=int,
        default=0,
        help="spectrogram delay for auto-detected bluetooth devices.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="loopback device sample rate",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="sample size in ms, higher values will decrease fps",
    )
    parser.add_argument(
        "--reference-max",
        type=int,
        default=3000,
        help="value used to tune maximum loudness of sound",
    )
    parser.add_argument(
        "--pipewire-fix",
        action="store_true",
        help="pipewire only, connect to output with custom loopback device. This prevents headsets from switching to 'handsfree' mode, which is mono and has lower audio quality. Sometimes this wont work unless sound is playing",
    )
    parser.add_argument(
        "--print-pipewire-node",
        action="store_true",
        help="will print all currently used pipewire nodes to monitor sound, then exit",
    )
    parser.add_argument(
        "--pipewire-node-id",
        type=str,
        default=None,
        help="ID of custom pipewire node to use. Set this to preferred node if spectroterm is launched before any soud is reproduced. Effective only whith --pipewire-fix. Use 'pw-list -o' to get list of available nodes, or use --print-pipewire-node",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s 0.6.0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    signal.signal(signal.SIGINT, sigint_handler)
    if args.print_pipewire_node:
        last_nodes = connect_pipewire(sc.default_speaker().id, only_get_name=True)
        for node in last_nodes:
            print(node)
        sys.exit()
    curses.wrapper(main, args)
