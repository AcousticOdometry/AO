# Installing RUNTIME_DEPENDENCIES requires at least CMake version 3.21
cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_API "Build Python API" OFF) # defined in `setup.py`

# Set CMake to use `vcpkg` to install dependencies. I avoids the need of
# passing it as an argument to `cmake`.
# https://github.com/microsoft/vcpkg/blob/master/docs/users/integration.md#using-an-environment-variable-instead-of-a-command-line-option
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
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
if(DEFINED CMAKE_TOOLCHAIN_FILE) # Print useful information
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

# ------------------------- Build the project library -------------------------
# TODO refactor into src/cpp/${PROJECT_NAME}/CMakeLists.txt
find_package(fmt CONFIG REQUIRED)

if(NOT BUILD_PYTHON_API) # Python API will not be built
    message(STATUS "Building library with libtorch")
    set(WITH_LIBTORCH TRUE)
    add_definitions(-DWITH_LIBTORCH)
    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
        message(STATUS "Torch global installation was not found")
        # TODO install if not found
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
            find_package (Python3 COMPONENTS Interpreter REQUIRED)
            message(STATUS "\
    Installing locally using ${CMAKE_SOURCE_DIR}/scripts/setup_libtorch.py"
            )
            execute_process(COMMAND 
                "${Python3_EXECUTABLE}" 
                "${CMAKE_SOURCE_DIR}/scripts/setup_libtorch.py"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
        endif()
        message(STATUS 
            "Using local installation at ${CMAKE_SOURCE_DIR}/libtorch"
        )
        list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
        find_package(Torch REQUIRED)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message(STATUS "Building library without libtorch")
endif()

# It is important that the library is static in order to build its Python
# wrapper
add_library(${PROJECT_NAME} STATIC 
    src/cpp/main.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
        src/cpp
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE
        fmt::fmt
        # Do not need to make this conditional, variables are empty
        "${TORCH_LIBRARIES}"
        "${TORCH_CUDA_LIBRARIES}"
)

if (WITH_LIBTORCH AND MSVC)
    # Pytorch: The following code block is suggested to be used on Windows.
    # According to https://github.com/pytorch/pytorch/issues/25457, the DLLs
    # need to be copied to avoid memory errors.
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

# ---------------------------- Build library tests ----------------------------
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ----------------------------- Build python api ------------------------------
if(BUILD_PYTHON_API)
    add_subdirectory(src/python)
endif()