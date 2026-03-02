cmake_minimum_required(VERSION 3.16)
project(mols-research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
find_package(Threads)

# LatinSquare library (sources under LatinSquare/)
set(LATIN_SQUARE_DIR ${CMAKE_SOURCE_DIR}/LatinSquare)
add_library(LatinSquare ${LATIN_SQUARE_DIR}/LatinSquare.cpp)
target_include_directories(LatinSquare PUBLIC ${LATIN_SQUARE_DIR})
target_link_libraries(LatinSquare PUBLIC Threads::Threads)

# All targets need src and LatinSquare on include path
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
target_include_directories(LatinSquare PUBLIC ${SRC_DIR})

# --- Serial executables ---
add_executable(generate_squares_serial
  ${SRC_DIR}/serial/GenerateSquares.cpp
)
target_link_libraries(generate_squares_serial PRIVATE LatinSquare)
target_include_directories(generate_squares_serial PRIVATE ${SRC_DIR} ${LATIN_SQUARE_DIR})

add_executable(check_squares_serial
  ${SRC_DIR}/serial/CheckMOLS.cpp
)
target_link_libraries(check_squares_serial PRIVATE LatinSquare)
target_include_directories(check_squares_serial PRIVATE ${SRC_DIR} ${LATIN_SQUARE_DIR})

# --- Utilities ---
add_executable(utilities ${SRC_DIR}/utils/Utils.cpp)
target_include_directories(utilities PRIVATE ${SRC_DIR}/utils)

# --- MPI (optional) ---
find_package(MPI)
if(MPI_CXX_FOUND)
  add_executable(generate_squares_mpi ${SRC_DIR}/mpi/GenerateSquares.cpp)
  target_link_libraries(generate_squares_mpi PRIVATE LatinSquare MPI::MPI_CXX)
  target_include_directories(generate_squares_mpi PRIVATE ${SRC_DIR} ${LATIN_SQUARE_DIR})
endif()

# --- CUDA (optional: -DBUILD_CUDA=ON when nvcc available) ---
option(BUILD_CUDA "Build CUDA targets" OFF)
if(BUILD_CUDA)
  enable_language(CUDA)
  add_executable(generate_squares_cuda
    ${SRC_DIR}/cuda/GenerateSquares.cu
  )
  target_link_libraries(generate_squares_cuda PRIVATE LatinSquare)
  target_include_directories(generate_squares_cuda PRIVATE ${SRC_DIR} ${LATIN_SQUARE_DIR})
  set_target_properties(generate_squares_cuda PROPERTIES
    CUDA_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON
  )
endif(BUILD_CUDA)
