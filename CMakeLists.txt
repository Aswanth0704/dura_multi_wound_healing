cmake_minimum_required(VERSION 3.11)
project(dura_multi_wound_healing CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#---------------------------------------------------------------------------#
# Which driver (main) to build.
# Override with:  cmake -DMAIN_SRC=src/<other_main>.cpp ..
#---------------------------------------------------------------------------#
set(MAIN_SRC "src/cydindrical_dura_single_tissue_multiwound.cpp"
    CACHE STRING "Path to the driver source containing main()")
message(STATUS "Driver: ${MAIN_SRC}")

include_directories(include)
include_directories(meshing)

#---------------------------------------------------------------------------#
# Eigen.  We need the directory that CONTAINS the Eigen/ folder, because the
# sources do #include <Eigen/Dense>.
#   Negishi: /scratch/.../athani_wound_healing_simulations/Eigen  -> parent is ".."
#   macOS/Homebrew: /opt/homebrew/include/eigen3/Eigen
# Override with: cmake -DEIGEN_INCLUDE_DIR=/path/containing/Eigen ..
#---------------------------------------------------------------------------#
set(EIGEN_INCLUDE_DIR "" CACHE PATH "Directory containing the Eigen/ header folder")
if(NOT EIGEN_INCLUDE_DIR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Eigen/Core")
        set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../eigen/Eigen/Core")
        set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../eigen")
    elseif(EXISTS "/opt/homebrew/include/eigen3/Eigen/Core")
        set(EIGEN_INCLUDE_DIR "/opt/homebrew/include/eigen3")
    elseif(EXISTS "/usr/include/eigen3/Eigen/Core")
        set(EIGEN_INCLUDE_DIR "/usr/include/eigen3")
    endif()
endif()
if(NOT EIGEN_INCLUDE_DIR OR NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR
        "Could not locate Eigen. Pass -DEIGEN_INCLUDE_DIR=<dir containing Eigen/>.")
endif()
message(STATUS "Eigen include dir: ${EIGEN_INCLUDE_DIR}")
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

#---------------------------------------------------------------------------#
# Boost (header-only here: boost/algorithm/string.hpp).
#   Negishi: /scratch/.../athani_wound_healing_simulations/boost -> parent ".."
#---------------------------------------------------------------------------#
set(BOOST_INCLUDE_DIR "" CACHE PATH "Directory containing the boost/ header folder")
if(NOT BOOST_INCLUDE_DIR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../boost/algorithm/string.hpp")
        set(BOOST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../boost_1_71_0/boost/algorithm/string.hpp")
        set(BOOST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../boost_1_71_0")
    elseif(EXISTS "/opt/homebrew/include/boost/algorithm/string.hpp")
        set(BOOST_INCLUDE_DIR "/opt/homebrew/include")
    elseif(EXISTS "/usr/include/boost/algorithm/string.hpp")
        set(BOOST_INCLUDE_DIR "/usr/include")
    endif()
endif()
if(NOT BOOST_INCLUDE_DIR OR NOT EXISTS "${BOOST_INCLUDE_DIR}/boost/algorithm/string.hpp")
    message(FATAL_ERROR
        "Could not locate Boost headers. Pass -DBOOST_INCLUDE_DIR=<dir containing boost/>.")
endif()
message(STATUS "Boost include dir: ${BOOST_INCLUDE_DIR}")
include_directories(SYSTEM ${BOOST_INCLUDE_DIR})

#---------------------------------------------------------------------------#
# OpenMP
#---------------------------------------------------------------------------#
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP NOT found - build will be serial (pragmas ignored).")
endif()

#---------------------------------------------------------------------------#
# Optional MKL acceleration for Eigen.  Only enable when Intel BLAS and the MKL
# headers are really present; EIGEN_USE_MKL_ALL without mkl.h is a hard error.
#---------------------------------------------------------------------------#
option(USE_MKL "Try to enable Eigen's MKL backend" ON)
set(_extra_libs "")
if(USE_MKL)
    set(BLA_VENDOR Intel10_64lp)
    find_package(BLAS)
    if(BLAS_FOUND AND DEFINED ENV{MKLROOT} AND EXISTS "$ENV{MKLROOT}/include/mkl.h")
        message(STATUS "MKL found at $ENV{MKLROOT} - enabling EIGEN_USE_MKL_ALL")
        include_directories(SYSTEM "$ENV{MKLROOT}/include")
        add_definitions(-DEIGEN_USE_MKL_ALL -DMKL_LP64)
        find_package(LAPACK)
        set(_extra_libs ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    else()
        message(STATUS "MKL not usable - building without the MKL backend")
    endif()
endif()

#---------------------------------------------------------------------------#
# Target.  NOTE: do NOT add src/file_io_gmsh_double.cpp - it redefines
# readGmshMsh22doubleWound and loadNodeSet0Based from src/file_io_gmsh.cpp.
#---------------------------------------------------------------------------#
add_executable(woundcpp3D
        ${MAIN_SRC}
        src/solver.cpp
        src/wound.cpp
        src/local_solver.cpp
        src/element_functions.cpp
        src/file_io.cpp
        src/file_io_gmsh.cpp
        meshing/myMeshGenerator.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(woundcpp3D PRIVATE OpenMP::OpenMP_CXX)
endif()
if(_extra_libs)
    target_link_libraries(woundcpp3D PRIVATE ${_extra_libs})
endif()
target_compile_features(woundcpp3D PRIVATE cxx_std_17)

#---------------------------------------------------------------------------#
# Tests.  Header-only checks, no solver linkage needed.
#   make test_mechanosensing && ./test_mechanosensing
#   or:  ctest
#---------------------------------------------------------------------------#
enable_testing()
add_executable(test_mechanosensing EXCLUDE_FROM_ALL tests/test_mechanosensing.cpp)
target_compile_features(test_mechanosensing PRIVATE cxx_std_17)
add_test(NAME mechanosensing COMMAND test_mechanosensing)

add_executable(test_diffusivity EXCLUDE_FROM_ALL tests/test_diffusivity.cpp)
target_compile_features(test_diffusivity PRIVATE cxx_std_17)
add_test(NAME diffusivity COMMAND test_diffusivity)

add_custom_target(tests DEPENDS test_mechanosensing test_diffusivity)
