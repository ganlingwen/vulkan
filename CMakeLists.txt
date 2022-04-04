cmake_minimum_required(VERSION 3.23.0)
project(computer_graphics)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)


add_library(computer_graphics_application
  computer_graphics_application.cpp)
target_link_libraries(computer_graphics_application PUBLIC
  Vulkan::Vulkan
  glfw
)


add_executable(run_computer_graphics main.cpp)
target_link_libraries(run_computer_graphics PUBLIC
  computer_graphics_application
)
