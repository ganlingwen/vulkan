#pragma once

#include <optional>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace cg {
struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  bool isCompute() const {
    return graphics_family.has_value() && present_family.has_value();
  }
};

class ComputerGraphicsApplication {
 public:
  ComputerGraphicsApplication();
  ~ComputerGraphicsApplication();

  void run();

 private:
  void initWindow();
  void initVulkan();
  void createSurface();
  void pickPhysicalDevice();
  void createLogicalDevice();

  GLFWwindow* window_;
  VkInstance instance_;
  VkSurfaceKHR surface_;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  QueueFamilyIndices indices_;
  VkDevice device_;
  VkQueue graphics_queue_;
  VkQueue present_queue_;
};
} // namespace cg