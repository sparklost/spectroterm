import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib


def get_app_name():
    """Get app name from pyproject.toml"""
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        if "project" in data and "version" in data["project"]:
            return str(data["project"]["name"])
        print("App name not specified in pyproject.toml")
        sys.exit()
    print("pyproject.toml file not found")
    sys.exit()


def get_version_number():
    """Get version number from pyproject.toml"""
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        if "project" in data and "version" in data["project"]:
            return str(data["project"]["version"])
        print("Version not specified in pyproject.toml")
        sys.exit()
    print("pyproject.toml file not found")
    sys.exit()


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


def fprint(text, color_code="\033[1;35m", prepend=f"[{PKGNAME.capitalize()} Build Script]: "):
    """Print colored text prepended with text, default is light purple"""
    if USE_COLOR:
        print(f"{color_code}{prepend}{text}\033[0m")
    else:
        print(f"{prepend}{text}")


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
    fprint(f"Compiling cython code with {"clang" if clang else "gcc"}{("mingw") if mingw else ""}")
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


def build_with_nuitka(onedir, clang, mingw):
    """Build with nuitka"""
    pkgname = get_app_name()

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

    # options
    if clang:
        os.environ["CFLAGS"] = "-Wno-macro-redefined"

    # platform-specific
    if sys.platform == "linux":
        options = []
    elif sys.platform == "win32":
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
        "--clang",
        action="store_true",
        help="use clang when building with nuitka",
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
        help="use mingw instead msvc on windows, has no effect on Linux and macOS, or with --clang flag",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    if sys.platform not in ("linux", "win32", "darwin"):
        sys.exit(f"This platform is not supported: {sys.platform}")
    if not args.nocython:
        try:
            build_cython(args.clang, args.mingw)
        except Exception as e:
            print(f"Failed building cython extensions, error: {e}")
    if args.nuitka:
        build_with_nuitka(args.onedir, args.clang, args.mingw)
        sys.exit()
    else:
        build_with_pyinstaller(args.onedir)
        sys.exit()
