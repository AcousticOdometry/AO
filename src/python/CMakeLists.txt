message(STATUS "Build Python API")

if(SKBUILD)
    message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
    # Scikit-Build does not add your site-packages to the search path
    # automatically, so we need to add the pybind11 specific directory here.
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
                "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT
    )
    list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Take into account that this name should be consistent with the contents of
# the `__init__.py` file and when calling PYBIND11_MODULE() in the python API
# code. This could be avoided with some code generation but it would be too
# complicated for this simple example.
set(PROJECT_PYTHON_API_NAME _ao)

# This function behaves very much like CMake’s builtin `add_library`
pybind11_add_module(${PROJECT_PYTHON_API_NAME} 
    MODULE 
        _ao.cpp
        _extractor.cpp
)

target_link_libraries(${PROJECT_PYTHON_API_NAME} 
    PRIVATE
        ${PROJECT_NAME}
)

target_compile_definitions(${PROJECT_PYTHON_API_NAME} 
    PRIVATE 
        VERSION_INFO=${PROJECT_VERSION}
)

# Extract the `vcpkg` dependencies from `vckpkg.json` so we can install them
# together with the python API.
string(JSON VCPKG_DEPENDENCIES_LENGTH LENGTH ${vcpkg_json} "dependencies")
math(EXPR _LIMIT "${VCPKG_DEPENDENCIES_LENGTH} - 1")
foreach(_IDX RANGE 0 ${_LIMIT})
    string(JSON VCPKG_DEPENDENCY GET ${vcpkg_json} "dependencies" ${_IDX})
    list(APPEND VCPKG_DEPENDENCIES ${VCPKG_DEPENDENCY})
endforeach()

# Install the `vcpkg` dependencies that are used in our ${PROJECT_NAME} library
# in the same DESTINATION as the python API. Google Test and `pybind11` are
# `vcpkg` dependencies but they are not used at runtime, only when building and
# therefore they won't be installed. Note that it only supports collecting the
# runtime dependencies for Windows, Linux and macOS platforms.
install(TARGETS ${PROJECT_PYTHON_API_NAME}
    RUNTIME_DEPENDENCIES 
        PRE_INCLUDE_REGEXES ${VCPKG_DEPENDENCIES}
        PRE_EXCLUDE_REGEXES ".*"
    DESTINATION .
)