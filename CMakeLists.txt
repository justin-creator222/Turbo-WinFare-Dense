cmake_minimum_required(VERSION 3.20)
project(turbo-dense LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for MinGW GCC
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -O3)
    add_link_options(-static-libgcc -static-libstdc++)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include
)

# Vulkan and System Libraries
set(VULKAN_LIB "C:/Windows/System32/vulkan-1.dll")
set(SYSTEM_LIBS ws2_32 bcrypt)

# Core Library
add_library(g4dense STATIC
    src/format.cpp
    src/manifest.cpp
    src/tokenizer.cpp
    src/sampling.cpp
    src/detokenizer.cpp
    src/http.cpp
    src/c_api.cpp
    src/cpu_reference.cpp
    src/vk_context.cpp
    src/vk_pipeline.cpp
    src/streamer.cpp
    src/draft_runtime.cpp
    src/speculator.cpp
    src/kv_cache.cpp
    src/runner.cpp
    src/telemetry.cpp
    src/openai_api.cpp
    src/server.cpp
)
target_link_libraries(g4dense PUBLIC ${SYSTEM_LIBS} ${VULKAN_LIB})

# Main CLI Executable
add_executable(turbo-dense src/main.cpp)
target_link_libraries(turbo-dense PRIVATE g4dense ${VULKAN_LIB})

# Testing
enable_testing()

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

add_executable(run_e2e_test tests/test_e2e_engine.cpp)
target_link_libraries(run_e2e_test PRIVATE g4dense ${VULKAN_LIB} ws2_32)
add_test(NAME run_e2e_test COMMAND $<TARGET_FILE:run_e2e_test>)
