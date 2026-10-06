cmake_minimum_required(VERSION 3.22)

project(PaulTrapSimulation VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ANALYSIS_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)

find_package(ROOT 6.24 REQUIRED COMPONENTS Core RIO Tree Hist Graf Gpad ROOTDataFrame)

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
    target_link_libraries(${target} PRIVATE
        analysis_options
        ROOT::Core
        ROOT::RIO
        ROOT::Tree
        ROOT::Hist
        ROOT::Graf
        ROOT::Gpad
        ROOT::ROOTDataFrame
    )
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# Add simulation programs explicitly after their interfaces and validation
# requirements have been agreed in docs/simulation/.
