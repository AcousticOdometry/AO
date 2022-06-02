# Installing RUNTIME_DEPENDENCIES requires at least CMake version 3.21
cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)

# Set CMake to use `vcpkg` to install dependencies. I avoids the need of
# passing it as an argument to `cmake`.
# https://github.com/microsoft/vcpkg/blob/master/docs/users/integration.md#using-an-environment-variable-instead-of-a-command-line-option
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE 
            "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING ""
        )
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE 
            "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING ""
        )
    endif()
endif()
if (DEFINED CMAKE_TOOLCHAIN_FILE) # Print useful information
    message(STATUS "Using toolchain: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "\
Not using `vcpkg`. Consider adding it to manage \
dependencies https://github.com/microsoft/vcpkg.git"
    )
endif()

# In order to avoid specifying package name and version in multiple files, we
# will use `vcpkg.json` in the repository root as reference and extract the
# apropiate variables from there.
file(READ ${CMAKE_SOURCE_DIR}/vcpkg.json vcpkg_json)
string(JSON PROJECT_NAME GET ${vcpkg_json} "name")
string(JSON VERSION_STRING GET ${vcpkg_json} "version-string")

project(${PROJECT_NAME} VERSION ${VERSION_STRING})

# It is recommended to split this file into multiple files and use
# `add_submodule` if it starts to grow too large.

# ------------------------- Build the project library -------------------------
# Recomended refactor into src/${PROJECT_NAME}/CMakeLists.txt
find_package(fmt CONFIG REQUIRED)

# It is important that the library is static in order to build its Python
# wrapper
add_library(${PROJECT_NAME} STATIC 
    src/ao/main.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
        src/${PROJECT_NAME}
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE
        fmt::fmt
)

# ---------------------------- Build library tests ----------------------------
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ----------------------------- Build python api ------------------------------
# Recomended refactor into src/python/CMakeLists.txt

# Option for building the project Python API, `setup.py` defines this variable
option(BUILD_PYTHON_API "Build Python API" OFF)
if(BUILD_PYTHON_API)
    add_subdirectory(src/python)
endif()