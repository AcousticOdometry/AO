# Acoustic Odometry
A standalone C++ library with dependencies managed by
[vcpkg](https://github.com/microsoft/vcpkg) accessible through Python using
[pybind11](https://github.co/pybind/pybind11).


## Example usage

### Create a clean Python virtual environment
```
python -m venv venv
```
Activate it on Windows
```
.\venv\Scripts\activate
```
otherwise
```
source ./venv/bin/activate
```

### Install this project
```
pip install git+https://github.com/AcousticOdometry/AO.git
```
or
```
pip install git+ssh://git@github.com/AcousticOdometry/AO.git
```
It will take a while to build as it will build the C++ dependencies as well,
but it will work. It is definitely not the most optimal way of installing a
package as we are installing as well the `vcpkg` package manager and building
from source dependencies that might as well be installed on the system. But
this allows a fast development environment where adding or removing C++
dependencies should be easy.

## Setup
### Install the requirements
Install [vcpkg](https://github.com/microsoft/vcpkg) requirements with the
addition of `cmake` and Python. It could be summarized as:
- [git](https://git-scm.com/downloads)
- Build tools ([Visual
  Studio](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio)
  on Windows or `gcc` on Linux for example)
- [cmake](#cmake)
- Python. Make sure to have development tools installed (`python3.X-dev` on
  Linux, being `X` your version of Python).

If running on a clean linux environment (like a container or Windows Subsystem
for Linux) you will need to install some additional tools as it is stated in
`vcpkg`.
```
sudo apt-get install build-essential curl zip unzip tar pkg-config libssl-dev python3-dev
```
This library is additionally based on modern C++17 with some C++20 features.
Therefore it requires a C++ compiler that supports C++20. In linux, `gcc`
version 11 is recommended. Fortunately, [several versions of `gcc` can co-live
together](https://askubuntu.com/a/1028656).

```
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11
```

#### Git Large File System
Needed for the model files

https://git-lfs.github.com/

#### CMake
Follow the [official instructions](https://cmake.org/install/).

The required `cmake` version is quite high, if you are using a Linux
distribution and installing `cmake` from the repositories take into account
that they might not be updated to the latest version. However there are options
to [install the latest version of `cmake` from the command
line](https://askubuntu.com/a/865294).

Make sure that when you run `cmake --version` the output is `3.21` or higher.
The reason for this is that we are using some of the `3.21` features to install
runtime dependencies (managed with `vcpkg`) together with our project so they
are available to Python when using its API.

### Clone this repository with `vcpkg`

Cone this repository with `vcpkg` as a submodule and navigate into it.
```
git clone --recursive git@github.com:esdandreu/python-extension-cpp.git
cd python-extension-cpp
```

Bootstrap `vcpkg` in Windows. Make sure you have [installed the
prerequisites](https://github.com/microsoft/vcpkg).
```
.\vcpkg\bootstrap-vcpkg.bat
```

Or in Linux/MacOS. Make sure you have [installed developer
tools](https://github.com/microsoft/vcpkg)
```
./vcpkg/bootstrap-vcpkg.sh
```

## Building

### Build locally with CMake
Navigate to the root of the repository and create a build directory.
```
mkdir build
```

Configure `cmake` to use `vcpkg`.
```
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$pwd/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

Build the project.
```
cmake --build build
```
### Build locally with Python

It is recommended to use a [clean virtual
environment](#create-a-clean-python-virtual-environment).

`scikit-build` is required before running the installer, as it is the package
that takes care of the installation. The rest of dependencies will be installed
automatically.

```
pip install scikit-build
```

Install the repository. By adding `[test]` to our install command we can
install additionally the test dependencies.
```
pip install .[test]
```


## Testing

### Test the C++ library with Google Test

```
ctest --test-dir build/tests
```

### Test the python extension

```
pytest
```

## Development

We try to use the following style guide for `pybind11`
https://developer.lsst.io/pybind11/style.html

### Mount Google Drive in Windows Subsystem for Linux

```
sudo mount -t drvfs G: /mnt/g
```
