cmake_minimum_required(VERSION 3.15)
project(MarketMakerCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for High-Frequency Trading (HFT) logic
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /Oi)
else()
    add_compile_options(-O3 -march=native)
endif()

# --- AUTOMATIC PYBIND11 DOWNLOAD ---
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)
# -----------------------------------

# Find Python - specifically the one in your venv
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/order_book.cpp
    src/bindings.cpp
)

# Create Python module
pybind11_add_module(fast_orderbook ${SOURCES})

# Set output directory to your python folder
set_target_properties(fast_orderbook PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../python"
)