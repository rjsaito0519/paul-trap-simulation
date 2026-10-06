cmake_minimum_required(VERSION 3.22)

project(PaulTrapSimulation VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ANALYSIS_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)

execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE ANALYSIS_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT ANALYSIS_GIT_COMMIT)
    set(ANALYSIS_GIT_COMMIT "unknown")
endif()
execute_process(
    COMMAND git status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE ANALYSIS_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(ANALYSIS_GIT_STATUS)
    set(ANALYSIS_GIT_DIRTY "true")
else()
    set(ANALYSIS_GIT_DIRTY "false")
endif()

add_library(analysis_options INTERFACE)
target_compile_features(analysis_options INTERFACE cxx_std_17)
target_compile_definitions(analysis_options INTERFACE
    ANALYSIS_VERSION="${PROJECT_VERSION}"
    ANALYSIS_GIT_COMMIT="${ANALYSIS_GIT_COMMIT}"
    ANALYSIS_GIT_DIRTY="${ANALYSIS_GIT_DIRTY}"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(analysis_options INTERFACE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    if(ANALYSIS_ENABLE_SANITIZERS)
        target_compile_options(analysis_options INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer
        )
        target_link_options(analysis_options INTERFACE
            -fsanitize=address,undefined
        )
    endif()
endif()

function(add_analysis_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE analysis_options)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# Add simulation programs explicitly after their interfaces and validation
# requirements have been agreed in docs/simulation/.

find_package(OpenMP REQUIRED COMPONENTS CXX)
enable_testing()

# Floquet stability engine (docs/simulation/stability.md).
add_library(paultrap_floquet STATIC src/floquet.cpp src/trajectory.cpp)
target_include_directories(paultrap_floquet PUBLIC src)
target_link_libraries(paultrap_floquet PRIVATE analysis_options)

add_analysis_executable(floquet_scan src/floquet_scan.cpp)
target_link_libraries(floquet_scan PRIVATE paultrap_floquet OpenMP::OpenMP_CXX)

add_analysis_executable(floquet_selftest src/floquet_selftest.cpp)
target_link_libraries(floquet_selftest PRIVATE paultrap_floquet)
add_test(NAME floquet_selftest COMMAND floquet_selftest)

add_analysis_executable(floquet_boundary src/floquet_boundary.cpp)
target_link_libraries(floquet_boundary PRIVATE paultrap_floquet OpenMP::OpenMP_CXX)

# 3D trajectory engine (docs/simulation/stability.md, finite-time capture).
add_analysis_executable(trajectory_run src/trajectory_run.cpp)
target_link_libraries(trajectory_run PRIVATE paultrap_floquet)
