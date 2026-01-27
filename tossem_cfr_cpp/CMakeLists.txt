cmake_minimum_required(VERSION 3.16)
project(tossem_cfr_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(train_mccfr
  src/main.cpp
  src/game_state.cpp
  src/abstraction.cpp
  src/hand_eval.cpp
)

target_include_directories(train_mccfr PRIVATE src)

# High-performance build flags
if (MSVC)
  target_compile_options(train_mccfr PRIVATE /O2)
else()
  target_compile_options(train_mccfr PRIVATE -O3 -march=native -ffast-math -fno-exceptions)
endif()
