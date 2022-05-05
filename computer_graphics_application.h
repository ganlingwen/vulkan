#pragma once

#include <optional>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace cg {
struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  bool isComplete() const {
    return graphics_family.has_value() && present_family.has_value();
  }
};

struct PhysicalDevice {
  VkPhysicalDevice device = VK_NULL_HANDLE;
  QueueFamilyIndices indices;
};

struct LogicalDevice {
  VkDevice device;
  VkQueue graphics;
  VkQueue present;

  void destroy() {}
};

struct Swapchain {
  VkSwapchainKHR chain;
  VkFormat format;
  VkExtent2D extent;

  std::vector<VkImage> images;
  std::vector<VkImageView> views;
  std::vector<VkFramebuffer> buffers;

  void destroy(const VkDevice &device) {
    for (auto framebuffer : buffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    for (auto view : views) {
      vkDestroyImageView(device, view, nullptr);
    }
    vkDestroySwapchainKHR(device, chain, nullptr);
  }
};

class ComputerGraphicsApplication {
public:
  ComputerGraphicsApplication();
  ~ComputerGraphicsApplication();

  void run();

private:
  void drawFrame();
  void recordCommandBuffer(VkCommandBuffer command_buffer,
                           uint32_t image_index);

  GLFWwindow *window_;
  VkInstance instance_;
  VkSurfaceKHR surface_;

  PhysicalDevice physical_;
  LogicalDevice logical_;
  Swapchain swapchain_;

  VkRenderPass render_pass_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;
  VkCommandPool command_pool_;
  VkCommandBuffer command_buffer_;

  VkSemaphore image_available_semaphore_;
  VkSemaphore render_finished_semaphore_;
  VkFence in_flight_fence_;
};
} // namespace cg
