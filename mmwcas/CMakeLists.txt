cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)


find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")
include_directories(${XLA_DIR} $ENV{CONDA_PREFIX}/include)

find_package(nanobind CONFIG REQUIRED)


enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")


add_library(_aoa SHARED "src/mmwcas/ffi/aoa.cu")
set_target_properties(_aoa PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                CUDA_STANDARD 17)
target_link_libraries(_aoa CUDA::cufft)
install(TARGETS _aoa LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/ffi")