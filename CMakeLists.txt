cmake_minimum_required(VERSION 3.10.0)
project(iterative_registration)

# Enable C++17 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable extensions to keep compatible with standards
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(src)
include_directories(include)

find_package(PCL REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})
find_package(OpenMP REQUIRED)

find_package(Open3D REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

add_library(cc SHARED include/cc.hpp src/cc.cpp)
target_link_libraries(cc Open3D::Open3D)
target_include_directories(cc PUBLIC
  include
  ${EIGEN3_INCLUDE_DIR}
  ${OPENMP_INCLUDE_DIRS}
)

# new main library
add_library(iterative_SE3_registration SHARED src/iterative_SE3_registration.cpp)
target_link_libraries(iterative_SE3_registration Open3D::Open3D OpenMP::OpenMP_CXX ${PCL_LIBRARIES})
target_include_directories(iterative_SE3_registration PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OPENMP_INCLUDE_DIRS}
)

add_executable(run_registration_method examples/run_registration_method.cpp)
add_dependencies(run_registration_method iterative_SE3_registration)
target_link_libraries(run_registration_method iterative_SE3_registration)

add_executable(registration_example examples/registration_example.cpp)
add_dependencies(registration_example iterative_SE3_registration cc)
target_link_libraries(registration_example iterative_SE3_registration cc)

add_executable(registration_example_fgr examples/registration_example_fgr.cpp)
add_dependencies(registration_example_fgr iterative_SE3_registration cc)
target_link_libraries(registration_example_fgr iterative_SE3_registration cc)

add_executable(benchmark_synthetic examples/benchmark_synthetic.cpp)
add_dependencies(benchmark_synthetic iterative_SE3_registration cc)
target_link_libraries(benchmark_synthetic iterative_SE3_registration cc)

add_executable(benchmark_lounge examples/benchmark_lounge.cpp)
add_dependencies(benchmark_lounge iterative_SE3_registration cc)
target_link_libraries(benchmark_lounge iterative_SE3_registration cc)

add_executable(test_se3_pure examples/test_se3_pure.cpp)
add_dependencies(test_se3_pure iterative_SE3_registration cc)
target_link_libraries(test_se3_pure iterative_SE3_registration cc)

add_executable(benchmark_kitti examples/benchmark_kitti.cpp)
add_dependencies(benchmark_kitti iterative_SE3_registration cc)
target_link_libraries(benchmark_kitti iterative_SE3_registration cc)

add_executable(benchmark_extreme_noise_bunny examples/benchmark_extreme_noise_bunny.cpp)
add_dependencies(benchmark_extreme_noise_bunny iterative_SE3_registration cc)
target_link_libraries(benchmark_extreme_noise_bunny iterative_SE3_registration cc)

add_executable(create_and_save_reg_problem examples/create_and_save_reg_problem.cpp)
add_dependencies(create_and_save_reg_problem iterative_SE3_registration cc)
target_link_libraries(create_and_save_reg_problem iterative_SE3_registration cc)