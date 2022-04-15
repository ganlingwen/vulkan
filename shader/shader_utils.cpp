#include "shader/shader_utils.h"

#include <fstream>

namespace cg {
std::vector<char> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("file " + filename +" not found");
  }
  const auto length = file.tellg();
  std::vector<char> buffer(length);
  file.seekg(0);
  file.read(buffer.data(), length);
  return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code, VkDevice device) {
  VkShaderModuleCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = code.size();
  info.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shader_module;
  if (vkCreateShaderModule(device, &info, nullptr, &shader_module) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
  return shader_module;
}
}  // namespace cg
