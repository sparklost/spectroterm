# spectroterm
Curses based terminal spectrum analyzer for currently playing audio.


## Features
- Works with PipeWire and PulseAudio
- Graph axes
- 3 color mode
- Bars slowly fall down
- Peak markers remain after bars fall down
- Custom frequency range
- Custom loudness range
- Custom colors
- Custom characters
- Custom fall speed and peak hold
- Automatic resizing  
- Fix for PipeWire switching headset to 'handsfree'


## Usage
```
usage: spectroterm [-h] [-a] [-b] [-c] [-p] [-f FALL_SPEED] [-o PEAK_HOLD] [-r BAR_CHARACTER]
                   [-k PEAK_CHARACTER] [--min-freq MIN_FREQ] [--max-freq MAX_FREQ]
                   [--min-db MIN_DB] [--max-db MAX_DB] [--green GREEN] [--orange ORANGE]
                   [--red RED] [--delay DELAY] [--bt-delay BT_DELAY] [--sample-rate SAMPLE_RATE]
                   [--sample-size SAMPLE_SIZE] [--reference-max REFERENCE_MAX] [--pipewire-fix]
                   [--print-pipewire-node] [--pipewire-node-id PIPEWIRE_NODE_ID] [-v]

Curses based terminal spectrum analyzer for currently playing audio

options:
  -h, --help            show this help message and exit
  -a, --axes            draw graph axes
  -b, --box             draw lines at terminal borders
  -c, --color           3 color mode
  -p, --peaks           draw peaks that disappear after some time
  -f, --fall-speed FALL_SPEED
                        speed at which bars fall in characters per second
  -o, --peak-hold PEAK_HOLD
                        time after which peak will dissapear, in ms
  -r, --bar-character BAR_CHARACTER
                        character used to draw bars
  -k, --peak-character PEAK_CHARACTER
                        character used to draw peaks
  --min-freq MIN_FREQ   minimum frequency on spectrum graph (x-axis)
  --max-freq MAX_FREQ   maximum frequency on spectrum graph (x-axis)
  --min-db MIN_DB       minimum loudness on spectrum graph (y-axis)
  --max-db MAX_DB       maximum loudness on spectrum graph (y-axis)
  --green GREEN         8bit ANSI color code for green part of bar
  --orange ORANGE       8bit ANSI color code for orange part of bar
  --red RED             8bit ANSI color code for red part of bar
  --delay DELAY         spectrogram delay for a better sync with sound.
  --bt-delay BT_DELAY   spectrogram delay for auto-detected bluetooth devices.
  --sample-rate SAMPLE_RATE
                        loopback device sample rate
  --sample-size SAMPLE_SIZE
                        sample size in ms, higher values will decrease fps
  --reference-max REFERENCE_MAX
                        value used to tune maximum loudness of sound
  --pipewire-fix        pipewire only, connect to output with custom loopback device. This
                        prevents headsets from switching to 'handsfree' mode, which is mono and
                        has lower audio quality. Sometimes this wont work unless sound is playing
  --print-pipewire-node
                        will print all currently used pipewire nodes to monitor sound, then exit
  --pipewire-node-id PIPEWIRE_NODE_ID
                        ID of custom pipewire node to use. Set this to preferred node if
                        spectroterm is launched before any soud is reproduced. Effective only
                        whith --pipewire-fix. Use 'pw-list -o' to get list of available nodes, or
                        use --print-pipewire-node
  -v, --version         show program's version number and exit
```

### Colors
Colors are provided as integer and they are [8bit ANSI color codes](https://gist.github.com/ConnerWill/d4b6c776b509add763e17f9f113fd25b#256-colors). -1 is default terminal color.


## Installing
- Pre-built binaries are available in releases
- From AUR: `yay -S spectroterm`
- Build, then copy built executable to system:  
`sudo cp dist/spectroterm /usr/local/sbin/`


## Building
1. Clone this repository
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
3. `cd spectroterm`
4. run build script: `uv run build.py`  
5. To build with Nuitka, add `--nuitka` flag. More optimized, smaller executable, long compile time. See [Nuitka](#nuitka) for more info.  


## Nuitka
To enable building with Nuitka, add `--nuitka` flag (takes a long time).  
Nuitka built binaries are more optimized.
Optionally, add `--clang` flag to tell nuitka to compile using llvm, which might run even faster.  
Nuitka requirements:
- on Linux: GCC or clang and `patchelf` package
- on Windows: [Visual Studio 2022](https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx) or mingw (will be downloaded by nuitka)
- on macOS install XCode via Apple Store


## Screenshots
![spectroterm screenshot 01](https://raw.githubusercontent.com/sparklost/spectroterm/refs/heads/main/.github/screenshots/01.png)
Spectroterm running in tmux:  
![spectroterm screenshot 02](https://raw.githubusercontent.com/sparklost/spectroterm/refs/heads/main/.github/screenshots/02.png)  
(Command: `spectroterm -pc -r ┃ --min-db=-70 --max-db=-20`)
