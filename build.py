import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib

PYTHON_MAX_MINOR = 13

CUSTOM_CFLAGS = [
    "-DNDEBUG",
    "-g0",
    "-O3",
    "-march=x86-64",
    "-mtune=generic",
    "-fno-semantic-interposition",
    "-fno-strict-overflow",
    "-fvisibility=hidden",
    # "-flto=thin",
]
CUSTOM_CXXFLAGS = CUSTOM_CFLAGS
CUSTOM_LDFLAGS = [
    "-Wl,-s",
    "-Wl,-O1",
    "-Wl,--sort-common",
    "-Wl,--as-needed",
    "-Wl,-z,pack-relative-relocs",
    "-Wl,--exclude-libs,ALL",
    # "-flto=thin",
]
CFLAGS_OLD = os.environ.get("CFLAGS", "")
CXXFLAGS_OLD = os.environ.get("CFLAGS", "")
LDFLAGS_OLD = os.environ.get("CFLAGS", "")


def get_app_name():
    """Get app name from pyproject.toml"""
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        if "project" in data and "version" in data["project"]:
            return str(data["project"]["name"])
        print("App name not specified in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    print("pyproject.toml file not found", file=sys.stderr)
    sys.exit(1)


def get_version_number():
    """Get version number from pyproject.toml"""
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        if "project" in data and "version" in data["project"]:
            return str(data["project"]["version"])
        print("Version not specified in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    print("pyproject.toml file not found", file=sys.stderr)
    sys.exit(1)


def is_gil_enabled():
    """Safely check if GIL is enabled"""
    try:
        return sys._is_gil_enabled()
    except AttributeError:
        return True


def get_python_version():
    """Get python major and minor versions"""
    if shutil.which("uv"):
        try:
            version_result = subprocess.run(["uv", "run", "--no-sync", "python", "-VV"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"uv error: {e}", file=sys.stderr)
            return sys.version_info.major, sys.version_info.minor, is_gil_enabled()
        all_parts = version_result.stdout.strip().split(" ")
        version_parts = all_parts[1].split(".")
        if len(version_parts) < 2:
            return sys.version_info.major, sys.version_info.minor, is_gil_enabled()
        return int(version_parts[0]), int(version_parts[1]), "free-threading" in all_parts[2]
    return sys.version_info.major, sys.version_info.minor, is_gil_enabled()


def supports_color():
    """Return True if the running terminal supports ANSI colors."""
    if sys.platform == "win32":
        return (os.getenv("ANSICON") is not None or
            os.getenv("WT_SESSION") is not None or
            os.getenv("TERM_PROGRAM") == "vscode" or
            os.getenv("TERM") in ("xterm", "xterm-color", "xterm-256color")
        )
    if not sys.stdout.isatty():
        return False
    return os.getenv("TERM", "") != "dumb"


PKGNAME = get_app_name()
PKGVER = get_version_number()
USE_COLOR = supports_color()


def fprint(text, color_code="\033[1;35m", prefix=f"[{PKGNAME.capitalize()} Build Script]: "):
    """Print colored text prefixed with text, default is light purple"""
    if USE_COLOR:
        print(f"{color_code}{prefix}{text}\033[0m")
    else:
        print(f"{prefix}{text}")


def check_python():
    """Check python version and print warning, and return True if runing inside pure python (no uv)"""
    if sys.version_info.major != 3:
        print(f"Python {sys.version_info.major} is not supported. Only Python 3 is supported.", file=sys.stderr)
        sys.exit(1)

    if os.environ.get("UV", ""):
        if sys.version_info.minor < 12 or sys.version_info.minor > PYTHON_MAX_MINOR:
            fprint(f'WARNING: Python {sys.version_info.major}.{sys.version_info.minor} is not supported but build may succeed. Run "python build.py" to let uv download and setup recommended temporary python interpreter.', color_code="\033[1;31m")
        else:
            try:
                version = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
                fprint(f"Using {version.stdout.strip()}")
            except Exception:
                pass
            fprint(f"Using Python {sys.version}")
        if not is_gil_enabled():
            fprint("WARNING: Freethreaded build may fail or built binary may crash.", color_code="\033[1;31m")
        return False

    try:
        version = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"uv error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("uv command not found, please ensure uv is installed and in PATH", file=sys.stderr)
        sys.exit(1)
    return True


def ensure_python():
    """Check current python and download correct python if needed"""
    _, minor, _ = get_python_version()
    if minor == PYTHON_MAX_MINOR:
        return None

    version = f"3.{PYTHON_MAX_MINOR}"
    # ensure there is no same-name freethreaded python
    subprocess.run(["uv", "python", "uninstall", f"3.{minor}+freethreaded"], check=False)

    fprint(f"Setting up python {version} for this project")
    subprocess.run(["uv", "python", "install", version], check=True)

    return version


def check_dev():
    """Check if its dev environment and set it up"""
    if importlib.util.find_spec("PyInstaller") is None or importlib.util.find_spec("nuitka") is None:
        subprocess.run(["uv", "sync", "--group", "build"], check=True)


def find_file_in_venv(lib_name, file_name):
    """Search for file in specified library in current venv"""
    if isinstance(file_name, list):
        file_name = os.path.join(*file_name)
    for root, dirs, files in os.walk(".venv"):
        if lib_name in dirs:
            lib_dir = os.path.join(root, lib_name)
            path = os.path.join(lib_dir, file_name)
            if os.path.isfile(path):
                return path
    else:
        fprint(f"{lib_name}/{file_name} not found")
        return


def patch_soundcard():
    """
    Search for soundcard/mediafoundation.py in .venv
    Prepend "if _ole32: " to "_ole32.CoUninitialize()" line while respecting indentation
    Search for soundcard/pulseaudio.py in .venv
    replace assert with proper exception
    """
    fprint("Patching soundcard")
    if not os.path.exists(".venv"):
        print(".venv dir not found")
        return

    # patch mediafoundation.py
    path = find_file_in_venv("soundcard", "mediafoundation.py")
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pattern = re.compile(r"^(\s*)_ole32\.CoUninitialize\(\)")
    changed = False
    for num, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            indent = match.group(1)
            lines[num] = f"{indent}if _ole32: _ole32.CoUninitialize()\n"
            changed = True
            break

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Patched file: {path}")
    else:
        print(f"Nothing to patch in file {path}")

    # patch pulseaudio.py
    path = find_file_in_venv("soundcard", "pulseaudio.py")
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pattern = re.compile(r"^(\s*)assert self\._pa_context_get_state")
    changed = False
    for num, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            indent = match.group(1)
            lines[num] = f"{indent}if self._pa_context_get_state(self.context) != _pa.PA_CONTEXT_READY:\n"
            lines.insert(num+1, f'{indent+"    "}raise RuntimeError("PulseAudio context not ready (no sound system?)")\n')
            changed = True
            break

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Patched file: {path}")
    else:
        print(f"Nothing to patch in file {path}")


def setup_compiler(clang, clear=False, overwrite=False, cflags=[], ldflags=[], cxxflags=[]):
    """Set compiler and its flags in environment variables"""
    if clang:
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
        os.environ["LD"] = "lld"
    if clear:
        os.environ["CFLAGS"] = CFLAGS_OLD
        os.environ["CXXFLAGS"] = CXXFLAGS_OLD
        os.environ["LDFLAGS"] = LDFLAGS_OLD
        return [], [], []
    cflags = ([] if overwrite else CFLAGS_OLD.split(" ")) + CUSTOM_CFLAGS + cflags
    cxxflags = ([] if overwrite else CXXFLAGS_OLD.split(" ")) + CUSTOM_CXXFLAGS + cxxflags
    ldflags = ([] if overwrite else LDFLAGS_OLD.split(" ")) + CUSTOM_LDFLAGS + ldflags
    if shutil.which("lld") and clang:
        ldflags.append("-fuse-ld=lld")
    os.environ["CFLAGS"] = " ".join(cflags)
    os.environ["CXXFLAGS"] = " ".join(cxxflags)
    os.environ["LDFLAGS"] = " ".join(ldflags)
    return cflags, cxxflags, ldflags


def build_numpy_lite(clang):
    """Build numpy without openblass to reduce final binary size"""
    if sys.platform != "linux":
        fprint("Skipping numpy lite (no openblas) building on non-linux platforms")
        return

    # check if numpy without blas is not already installed
    cmd = [
        "uv", "run", "python", "-c",
        "import numpy; print(int(numpy.__config__.show_config('dicts')['Build Dependencies']['blas'].get('found', False)))",
    ]
    if int(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()):
        fprint("Building numpy lite (no openblas)")
        if clang:
            os.environ["CC"] = "clang"
            os.environ["CXX"] = "clang++"
        subprocess.run(["uv", "pip", "install", "pip"], check=True)   # because uv wont work with --config-settings as intended
        try:
            if sys.platform == "win32":
                python_interpreter = r".venv\Scripts\python.exe"
            else:
                python_interpreter = ".venv/bin/python"
            subprocess.run([python_interpreter, "-m", "pip", "uninstall", "--yes", "numpy"], check=True)
            subprocess.run([
                python_interpreter, "-m", "pip", "install", "--no-cache-dir", "--no-binary=:all:", "numpy",
                "--config-settings=setup-args=-Dblas=None",
                "--config-settings=setup-args=-Dlapack=None",
            ], check=True)
        except subprocess.CalledProcessError:   # fallback
            fprint("Failed building numpy lite (no openblas), faling back to default numpy")
            subprocess.run(["uv", "pip", "install", "numpy"], check=True)
        subprocess.run(["uv", "pip", "uninstall", "pip"], check=True)
    else:
        fprint("Numpy lite (no openblas) is already built")


def build_cython(clang, mingw):
    """Build cython extensions"""
    clang = clang or os.environ.get("CC") == "clang"
    fprint(f"Compiling cython code with {"clang" if clang else "gcc"}{("mingw") if mingw else ""}")
    setup_compiler(clang)
    cmd = ["uv", "run", "python", "setup.py", "build_ext", "--inplace"]
    if clang:
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
    elif mingw and sys.platform == "win32":
        cmd.append("--compiler=mingw32")   # covers mingw 32 and 64

    # run process with control of stdout
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        line_clean = line.rstrip("\n")
        if len(line_clean) < 100 and not any(s in line_clean for s in ("Cythonizing", "Compiling", "creating", "  warn(")):
            print(line_clean)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    os.remove("spectrum_cython.c")
    shutil.rmtree("build")


def build_with_pyinstaller(onedir):
    """Build with pyinstaller"""
    pkgname = get_app_name()
    mode = "--onedir" if onedir else "--onefile"
    hidden_imports = ["--hidden-import=pyfftw"]
    exclude_imports = ["--exclude-module=cython"]
    package_data = []

    # platform-specific
    if sys.platform == "linux":
        options = []
    elif sys.platform == "win32":
        options = ["--console"]
    elif sys.platform == "darwin":
        options = []


    # prepare command and run it
    cmd = [
        "uv", "run", "python", "-m", "PyInstaller",
        mode,
        *hidden_imports,
        *exclude_imports,
        *package_data,
        *options,
        "--noconfirm",
        "--clean",
        f"--name={pkgname}",
        "main.py",
    ]
    cmd = [arg for arg in cmd if arg != ""]
    fprint("Starting pyinstaller")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(e.returncode)

    # cleanup
    fprint("Cleaning up")
    try:
        os.remove(f"{pkgname}.spec")
        shutil.rmtree("build")
    except FileNotFoundError:
        pass
    fprint(f"Finished building {pkgname}")


def build_with_nuitka(onedir, clang, mingw, print_cmd=False):
    """Build with nuitka"""
    clang = clang or os.environ.get("CC") == "clang"
    pkgname = get_app_name()

    if not print_cmd:
        build_numpy_lite(clang)

    mode = "--standalone" if onedir else "--onefile"
    compiler = ""
    if clang:
        compiler = "--clang"
    elif mingw:
        compiler = "--mingw64"
    python_flags = ["--python-flag=-OO"]
    hidden_imports = ["--include-module=pyfftw"]
    exclude_imports = ["--nofollow-import-to=cython"]
    package_data = ["--include-package-data=soundcard"]

    setup_compiler(clang)

    # options
    if clang:
        os.environ["CFLAGS"] = "-Wno-macro-redefined"

    # platform-specific
    if sys.platform == "linux":
        options = []
    elif sys.platform == "win32" and not print_cmd:
        patch_soundcard()
        options = ["--assume-yes-for-downloads"]
    elif sys.platform == "darwin":
        options = [
            f"--macos-app-name={get_app_name()}",
            f"--macos-app-version={get_version_number()}",
            "--macos-app-protected-resource=NSMicrophoneUsageDescription:Microphone access for recording voice message.",
        ]

    # prepare command and run it
    cmd = [
        "uv", "run", "python", "-m", "nuitka",
        mode,
        compiler,
        *python_flags,
        *hidden_imports,
        *exclude_imports,
        *package_data,
        *options,
        "--lto=yes",
        "--remove-output",
        "--output-dir=dist",
        f"--output-filename={pkgname}",
        "main.py",
    ]
    cmd = [arg for arg in cmd if arg != ""]
    if print_cmd:
        print(" ".join(cmd))
        sys.exit(0)
    fprint("Starting nuitka")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(e.returncode)

    # cleanup
    fprint("Cleaning up")
    try:
        os.remove(f"{pkgname}.spec")
        shutil.rmtree("build")
    except FileNotFoundError:
        pass
    fprint(f"Finished building {pkgname}")


def parser():
    """Setup argument parser for CLI"""
    parser = argparse.ArgumentParser(
        prog="build.py",
        description=f"build script for {PKGNAME}",
    )
    parser._positionals.title = "arguments"
    parser.add_argument(
        "--nuitka",
        action="store_true",
        help="build with nuitka, takes a long time, but more optimized executable",
    )
    parser.add_argument(
        "--noclang",
        action="store_true",
        help="script prefers clang if its installed, set this to not use it, or change CC and LD env vars",
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="build into directory instead single executable",
    )
    parser.add_argument(
        "--nocython",
        action="store_true",
        help="build without compiling cython code",
    )
    parser.add_argument(
        "--mingw",
        action="store_true",
        help="use mingw instead msvc on windows, has no effect on Linux and macOS",
    )
    parser.add_argument(
        "--nobuild",
        action="store_true",
        help="only configure environment, but dont build endcord",
    )
    parser.add_argument(
        "--print-cmd",
        action="store_true",
        help="print build command for nuitka or pyinstaller and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    clang = not (args.noclang or args.mingw)

    if args.print_cmd:
        if args.nuitka:
            build_with_nuitka(args.onedir, clang, args.mingw, args.nosoundcard, print_cmd=True)
        else:
            build_with_pyinstaller(args.onedir, args.nosoundcard, print_cmd=True)
        sys.exit(0)

    if check_python():
        version = ensure_python()
        if version:
            os.execvp("uv", ["uv", "run", "-p", version, *sys.argv])
        else:
            os.execvp("uv", ["uv", "run", *sys.argv])
        sys.exit(0)

    if args.nobuild:
        sys.exit(0)

    if sys.platform not in ("linux", "win32", "darwin"):
        print(f"This platform is not supported: {sys.platform}", file=sys.stderr)
        sys.exit(1)

    check_dev()
    if not args.nocython:
        try:
            build_cython(clang, args.mingw)
        except Exception as e:
            print(f"Failed building cython extensions, error: {e}")
    if args.nuitka:
        build_with_nuitka(args.onedir, clang, args.mingw)
    else:
        build_with_pyinstaller(args.onedir)

    sys.exit(0)
