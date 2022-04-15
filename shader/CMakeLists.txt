add_library(shader_utils
  shader_utils.cpp)
target_include_directories(shader_utils PUBLIC
  ..
)
target_link_libraries(shader_utils PUBLIC
  Vulkan::Vulkan
  glfw
)
