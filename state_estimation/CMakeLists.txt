cmake_minimum_required(VERSION 3.16)

project(state_estimation
  VERSION 0.1.0
  LANGUAGES CXX
)

add_library(state_estimation
  src/OrientationEstimator.cpp
  src/SimulatedImu.cpp
)

target_include_directories(state_estimation
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(state_estimation PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(state_estimation PRIVATE /W4)
else()
  target_compile_options(state_estimation PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(state_estimation_demo
  examples/demo.cpp
)
target_link_libraries(state_estimation_demo PRIVATE state_estimation)

include(CTest)
if(BUILD_TESTING)
  add_executable(state_estimation_tests
    tests/orientation_estimator_tests.cpp
  )
  target_link_libraries(state_estimation_tests PRIVATE state_estimation)
  add_test(NAME state_estimation_tests COMMAND state_estimation_tests)
endif()

