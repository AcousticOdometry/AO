"""Build setup

Setup the build environment for a C++ python extension module with `vcpkg` as
C++ package manager. It uses `scikit-build` to build the C++ extension module
and `GitPython` to pull `vcpkg` when it is defined as a submodule (as it is
recommended).

It is recommended to set the package metadata in the file `pyproject.toml`
minimizing the amount of package specific configuration in this file.

Raises:
    RuntimeError: If `vcpkg` is not a submodule of the repository.
"""

import json
import sys

from pathlib import Path
from shutil import rmtree
from setuptools import find_packages

try:
    from skbuild import setup
except ImportError as e:
    print(
        "Please update pip, you need pip 10 or greater,\n or you need to "
        "install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise e

PROJECT_SOURCE_DIR = Path(__file__).parent

# For some reason, running this file twice in a row causes the build to fail:
# fatal error C1083: Cannot open include file: 'io.h'
# Therefore the workaround is to clean the `_skbuild` directory before running
SKBUILD_DIR = PROJECT_SOURCE_DIR / "_skbuild"
if SKBUILD_DIR.exists():
    print(f'Removing previous installation: {SKBUILD_DIR}')
    rmtree(str(SKBUILD_DIR))

# Make sure that `vcpkg` package manager is available as a submodule
VCPKG_DIR = PROJECT_SOURCE_DIR / "vcpkg"
VCPKG_CMAKE_TOOLCHAIN = VCPKG_DIR / "scripts" / "buildsystems" / "vcpkg.cmake"
if not VCPKG_CMAKE_TOOLCHAIN.is_file():
    # Update only `vcpkg` as a submodule of this repository.
    from git import Repo
    for submodule in  Repo(PROJECT_SOURCE_DIR).submodules:
        if submodule.name == "vcpkg":
            submodule.update(init=True)
            break
    else:
        raise RuntimeError("Could not find submodule `vcpkg`")



# In order to avoid specifying package name and version in multiple files, we
# will use `vcpkg.json` in the repository root as reference and extract the
# apropiate variables from there.
with open(PROJECT_SOURCE_DIR / "vcpkg.json") as f:
    vcpkg_json = json.load(f)
    # Required
    PROJECT_NAME = vcpkg_json["name"]
    PROJECT_VERSION_STRING = vcpkg_json["version-string"]

# scikit-build will take care of puting our compiled C++ library together with
# our python package so it can access it. The name of the python package will
# be determined by the name of the folder that contains an `__init__.py` file.
# In this repository, python packages must be placed under path defined by
# `python_packages_root`.
python_packages_root = "src/python"
packages = find_packages(python_packages_root)
# ! The only package with access to the compiled C++ library is `packages[0]`,
# ! other packages have no direct access to compiled C++ code.

setup(
    # Package metadata, comment out if it is provided in `pyproject.toml`.
    # name=PROJECT_NAME, # Use the name defined in `vcpkg.json`
    version=PROJECT_VERSION_STRING, # Set version from `vcpkg.json`
    # Python package information is defined above
    packages=packages,
    package_dir={"": python_packages_root},
    cmake_install_dir=python_packages_root + "/" + packages[0],
    # CMake must be used allways, otherwise C++ dependencies won't be installed
    # ! setup_requires=["cmake"] should not be used, as it causes `vcpkg` to fail
    cmake_with_sdist=True, 
    # Signal cmake to use `vcpkg`
    cmake_args=[
        f"-DCMAKE_TOOLCHAIN_FILE:PATH={str(VCPKG_CMAKE_TOOLCHAIN.resolve())}",
        "-DBUILD_PYTHON_API=ON",
        "-DBUILD_TESTS=OFF",
        ],
    )
