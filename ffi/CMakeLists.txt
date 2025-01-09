cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)


find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")

add_library(_cuda_bp SHARED "src/jax_ffi_bp/cuda_bp.cu")
set_target_properties(_cuda_bp PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                CUDA_STANDARD 17)
target_include_directories(_cuda_bp PUBLIC ${XLA_DIR})
install(TARGETS _cuda_bp LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
