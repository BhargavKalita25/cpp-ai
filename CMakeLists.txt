cmake_minimum_required(VERSION 3.14)
project(cpp_ml_starter LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ml INTERFACE)
target_include_directories(ml INTERFACE include)

add_executable(ml_demo src/main.cpp)
target_link_libraries(ml_demo PRIVATE ml)

enable_testing()
add_executable(test_lr tests/test_lr.cpp)
target_link_libraries(test_lr PRIVATE ml)
add_test(NAME test_lr COMMAND test_lr)

add_executable(test_logreg tests/test_logreg.cpp)
target_link_libraries(test_logreg PRIVATE ml)
add_test(NAME test_logreg COMMAND test_logreg)