#pragma once

#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace cg {
std::vector<char> readFile(const std::string &filename);

VkShaderModule createShaderModule(const std::vector<char> &code,
                                  VkDevice device);

VkShaderModule createShaderModule(const std::string &source_filename,
                                  VkDevice device);
} // namespace cg
