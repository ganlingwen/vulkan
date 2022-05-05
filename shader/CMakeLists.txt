file(GLOB_RECURSE GLSL_SOURCE_FILES
  "*.frag"
  "*.vert"
)
foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND glslc ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL}
  )
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)


set(SHADER_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
configure_file(shader_utils.cpp.in shader_utils.cpp)


add_custom_target(shaders DEPENDS
  ${SPIRV_BINARY_FILES}
  shader_utils.cpp
)


add_library(shader_utils
  shader_utils.cpp)
target_include_directories(shader_utils PUBLIC
  ..
)
target_link_libraries(shader_utils PUBLIC
  Vulkan::Vulkan
  glfw
)
add_dependencies(shader_utils shaders)
