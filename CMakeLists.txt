cmake_minimum_required(VERSION 3.20)
project(TurboWinFareDense LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# System and compiler flags
if (MSVC)
    add_compile_options(/W4 /O2 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -O3)
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party/vulkan/include
)

# Link Libraries for Windows
set(SYSTEM_LIBS
    ws2_32
    bcrypt
    shell32
)

# Vulkan loader
find_library(VULKAN_LIB
    NAMES vulkan-1 vulkan
    PATHS
        ${PROJECT_SOURCE_DIR}/third_party/vulkan/lib
        $ENV{VULKAN_SDK}/Lib
        $ENV{VK_SDK_PATH}/Lib
)

if (NOT VULKAN_LIB)
    if (EXISTS "C:/Windows/System32/vulkan-1.dll")
        set(VULKAN_LIB "C:/Windows/System32/vulkan-1.dll")
    else()
        message(FATAL_ERROR "Vulkan loader library not found!")
    endif()
endif()

# Build-time SPIR-V shader compilation with DXC
set(DXC_EXE "${PROJECT_SOURCE_DIR}/build/dxc.exe")
file(GLOB HLSL_SHADERS "${PROJECT_SOURCE_DIR}/shaders/*.hlsl")
set(SPV_OUTPUTS "")
file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/build/shaders")

foreach(SHADER_SRC ${HLSL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME_WE)
    set(SPV_OUT "${PROJECT_SOURCE_DIR}/build/shaders/${SHADER_NAME}.spv")
    list(APPEND SPV_OUTPUTS ${SPV_OUT})
    add_custom_command(
        OUTPUT ${SPV_OUT}
        COMMAND ${DXC_EXE} -spirv -T cs_6_6 -E main -fspv-target-env=vulkan1.3 -Fo ${SPV_OUT} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC} "${PROJECT_SOURCE_DIR}/shaders/Common.hlsli"
        COMMENT "Compiling SPIR-V shader: ${SHADER_NAME}.hlsl -> ${SHADER_NAME}.spv"
        VERBATIM
    )
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPV_OUTPUTS})

# Copy GUI assets target
add_custom_target(copy_gui ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/gui"
        "${PROJECT_SOURCE_DIR}/build/gui"
    COMMENT "Syncing Web GUI assets"
)

# Core engine library
file(GLOB G4DENSE_SOURCES
    src/*.cpp
)

add_library(g4dense STATIC ${G4DENSE_SOURCES})
add_dependencies(g4dense compile_shaders copy_gui)
target_link_libraries(g4dense PUBLIC ${SYSTEM_LIBS} ${VULKAN_LIB})

# Main CLI Executable
add_executable(turbo-dense src/main.cpp)
add_dependencies(turbo-dense compile_shaders copy_gui)
target_link_libraries(turbo-dense PRIVATE g4dense ${VULKAN_LIB})

add_executable(run_turbo_dense src/main.cpp)
add_dependencies(run_turbo_dense compile_shaders copy_gui)
target_link_libraries(run_turbo_dense PRIVATE g4dense ${VULKAN_LIB})

# Testing Suite
enable_testing()

add_executable(run_contracts_test tests/test_contracts.cpp)
target_link_libraries(run_contracts_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_contracts_test COMMAND $<TARGET_FILE:run_contracts_test>)

add_executable(run_format_test tests/test_format.cpp)
target_link_libraries(run_format_test PRIVATE g4dense)
add_test(NAME run_format_test COMMAND $<TARGET_FILE:run_format_test>)

add_executable(run_tokenizer_test tests/test_tokenizer.cpp)
target_link_libraries(run_tokenizer_test PRIVATE g4dense)
add_test(NAME run_tokenizer_test COMMAND $<TARGET_FILE:run_tokenizer_test>)

add_executable(run_sampling_test tests/test_sampling.cpp)
target_link_libraries(run_sampling_test PRIVATE g4dense)
add_test(NAME run_sampling_test COMMAND $<TARGET_FILE:run_sampling_test>)

add_executable(run_cpu_reference_test tests/test_cpu_reference.cpp)
target_link_libraries(run_cpu_reference_test PRIVATE g4dense)
add_test(NAME run_cpu_reference_test COMMAND $<TARGET_FILE:run_cpu_reference_test>)

add_executable(run_vk_device_test tests/test_vk_device.cpp)
target_link_libraries(run_vk_device_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_vk_device_test COMMAND $<TARGET_FILE:run_vk_device_test>)

add_executable(run_gpu_kernels_test tests/test_gpu_kernels.cpp)
target_link_libraries(run_gpu_kernels_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_gpu_kernels_test COMMAND $<TARGET_FILE:run_gpu_kernels_test>)

add_executable(run_streamer_test tests/test_streamer.cpp)
target_link_libraries(run_streamer_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_streamer_test COMMAND $<TARGET_FILE:run_streamer_test>)

add_executable(run_smoke_engine_test tests/test_smoke_engine.cpp)
target_link_libraries(run_smoke_engine_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_smoke_engine_test COMMAND $<TARGET_FILE:run_smoke_engine_test>)

add_executable(run_speculative_test tests/test_speculative.cpp)
target_link_libraries(run_speculative_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_speculative_test COMMAND $<TARGET_FILE:run_speculative_test>)

# --- P1.3: tests that existed on disk but were never registered ------------------
#
# test_gpu_forward is the ONLY test that touches the real 31B container, and it was
# absent from this file -- so `ctest` was green while nothing exercised the real model.
# These run with WORKING_DIRECTORY at the source root: ctest otherwise runs them from
# build/, where `models/` and `tests/fixtures/` do not resolve, and their fs::exists
# guards turned that into a silent pass.

add_executable(run_detokenizer_test tests/test_detokenizer.cpp)
target_link_libraries(run_detokenizer_test PRIVATE g4dense)
add_test(NAME run_detokenizer_test COMMAND $<TARGET_FILE:run_detokenizer_test>
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(run_kv_cache_test tests/test_kv_cache.cpp)
target_link_libraries(run_kv_cache_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_kv_cache_test COMMAND $<TARGET_FILE:run_kv_cache_test>
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(run_gpu_forward_test tests/test_gpu_forward.cpp)
target_link_libraries(run_gpu_forward_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_gpu_forward_test COMMAND $<TARGET_FILE:run_gpu_forward_test>
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(run_gpu_forward_test PROPERTIES TIMEOUT 1800)

# P0.1 coherence check. Token count is deliberately small here so ctest stays tractable at
# the current ~23 s/token; judge coherence from a longer manual run.
add_executable(run_real_generation_test tests/test_real_generation.cpp)
target_link_libraries(run_real_generation_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_real_generation_test COMMAND $<TARGET_FILE:run_real_generation_test> 6
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(run_real_generation_test PROPERTIES TIMEOUT 3600)

# Existing tests that reach for repo-root resources need the same working directory.
set_tests_properties(run_smoke_engine_test run_speculative_test run_streamer_test
                     run_cpu_reference_test run_gpu_kernels_test
                     PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Smallest test that can detect a forward pass whose output does not vary with its input.
# The single-token oracle diff structurally cannot: it only ever runs position 0.
add_executable(run_context_dependence_test tests/test_context_dependence.cpp)
target_link_libraries(run_context_dependence_test PRIVATE g4dense ${VULKAN_LIB})
add_test(NAME run_context_dependence_test COMMAND $<TARGET_FILE:run_context_dependence_test>
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(run_context_dependence_test PROPERTIES TIMEOUT 1800)

# Prompt pipeline: string -> chat template -> tokens. No GPU, no model container.
add_executable(run_prompt_pipeline_test tests/test_prompt_pipeline.cpp)
target_link_libraries(run_prompt_pipeline_test PRIVATE g4dense)
add_test(NAME run_prompt_pipeline_test COMMAND $<TARGET_FILE:run_prompt_pipeline_test>
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Performance attribution harness. Not a test: it measures and prints, and is the instrument
# every optimization in the speed round is judged by.
add_executable(bench_gpu tools/bench_gpu.cpp)
target_link_libraries(bench_gpu PRIVATE g4dense ${VULKAN_LIB})
