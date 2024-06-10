cmake_minimum_required(VERSION 3.9)

project(
    cuFEM
    VERSION 0.0.1
    LANGUAGES C CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-options -Wall -Wextra -Werror -Wpedantic)

option(ENABLE_COVERAGE "Enable coverage" 1)
option(ENABLE_SANITIZERS "Enable sanitizers" 0)

if(ENABLE_COVERAGE OR ENABLE_SANITIZERS)
    add_compile_options(-g -O0)
elseif()
    add_compile_options(-Ofast)
endif()

if(ENABLE_COVERAGE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--coverage)
    add_link_options(--coverage)
endif()

if(ENABLE_SANITIZERS)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-fsanitize=address -fsanitize=leak -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=leak -fsanitize=undefined)
endif()

set(CUDA_FILES
    ${PROJECT_SOURCE_DIR}/src/CG.cu
)

set(CXX_FILES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

set(INCLUDE_FILES
    include/CG.hpp
    include/Timer.hpp
    include/Handler.hpp
    include/Vector.hpp
)

set(INCLUDE_LOCATIONS
    include
)

#include_directories(${INCLUDES})

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PUBLIC ${CUDA_FILES} ${CXX_FILES} ${INCLUDE_FILES}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${INCLUDE_LOCATIONS}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC cudart cusparse cublas
)
