import shutil
import platform
import requests
import tempfile

from pathlib import Path
from zipfile import ZipFile

SYSTEM = platform.system()

BINARIES = {
    'Windows': {
        'cpu': "libtorch-win-shared-with-deps-1.11.0%2Bcpu.zip",
        'cu113': "libtorch-win-shared-with-deps-1.11.0%2Bcu113.zip",
        },
    'Darwin': {
        'cpu': "libtorch-macos-1.11.0.zip",
        },
    'Linux': {
        'cpu': "libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip",
        'cu113': "libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip",
        }
    }


def setup_libtorch(force: bool = False, compute_platform: str = 'cpu'):
    root_dir = Path(__file__).parent.parent
    libtorch_path = (root_dir / 'libtorch').absolute()
    try:
        libtorch_path.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        if force:
            print(f"Overwriting {libtorch_path}")
            shutil.rmtree(libtorch_path)
        else:
            raise RuntimeError(
                f"{libtorch_path} already exists. Use --force to overwrite"
                )
    zip_name = BINARIES[SYSTEM][compute_platform]
    download_url = (
        "https://download.pytorch.org/libtorch/" +
        f"{compute_platform}/{zip_name}"
        )
    with tempfile.TemporaryFile() as f:
        print("Downloading libtorch...", end='\r')
        response = requests.get(download_url)
        print(f"Downloaded libtorch for {SYSTEM} {compute_platform}")
        f.write(response.content)
        with ZipFile(f) as zip_file:
            print("Unpacking libtorch...", end='\r')
            zip_file.extractall(root_dir)
            print(f"Unpacked libtorch to {libtorch_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Setup libtorch')
    # TODO check if there are binaries available for the current system
    parser.add_argument(
        'compute_platform',
        type=str,
        default='cpu',
        nargs='?',
        choices=list(BINARIES[SYSTEM].keys()),
        help="Compute platform to use",
        )
    parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help="Overwrite libtorch even if it is found",
        )
    args = parser.parse_args()
    setup_libtorch(**vars(args))