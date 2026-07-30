cmake_minimum_required(VERSION 3.11)
project(circularwoundmechanicscpp3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(include)
include_directories(meshing)
include_directories(src)
include_directories(../eigen)
include_directories(../boost_1_71_0)

set(BOOST_ROOT "../boost_1_71_0")
find_package(Boost)
IF (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
endif()
set (Boost_USE_STATIC_LIBS OFF) # enable dynamic linking
set (Boost_USE_MULTITHREAD ON)  # enable multithreading

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
IF (BLAS_FOUND)
    include_directories(${BLAS_INCLUDE_DIR})
    add_definitions(-DEIGEN_USE_MKL_ALL)
    add_definitions(-DMKL_LP64)
endif()
find_package(LAPACK)
IF (LAPACK_FOUND)
    include_directories(${LAPACK_INCLUDE_DIR})
endif()

add_executable(woundcpp3D
        include/myMeshGenerator.h
        include/solver.h
        include/wound.h
        include/local_solver.h
        include/element_functions.h
        include/file_io.h
        include/file_io_gmsh.h
        meshing/myMeshGenerator.cpp
        src/exact_dura_only_double_wound.cpp
        src/solver.cpp
        src/wound.cpp
        src/local_solver.cpp
        src/element_functions.cpp
        src/file_io_gmsh.cpp
        src/file_io.cpp)

target_link_libraries (woundcpp3D ${Boost_LIBRARIES} OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) #
target_compile_features(woundcpp3D PRIVATE cxx_std_17)
