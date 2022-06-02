message(STATUS "Build tests")
find_package(GTest CONFIG REQUIRED)
find_package(SndFile REQUIRED)

enable_testing()

# Tests can be run with `ctest` or by directly running the test executable
# that will be named as `${PROJECT_TESTS_NAME}`. The name of the executable
# is not really relevant as `ctest` will find it anyway thanks to Google
# Test.
set(PROJECT_TESTS_NAME test_${PROJECT_NAME})

file(GLOB TEST_FILES test_*.cpp)
add_executable(${PROJECT_TESTS_NAME} 
    ${TEST_FILES}
)

target_link_libraries(${PROJECT_TESTS_NAME} 
    PRIVATE
        ${PROJECT_NAME}
        GTest::gtest_main
        SndFile::sndfile
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_TESTS_NAME})