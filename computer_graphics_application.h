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
  void createSwapChain();
  void createImageViews();
  void createRenderPass();
  void createGraphicsPipeline();
  void createFramebuffers();
  
  void createCommandPool();
  void createCommandBuffer();
  void recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index);
  void createSyncObjects();
  void drawFrame();

  GLFWwindow* window_;
  VkInstance instance_;
  VkSurfaceKHR surface_;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  QueueFamilyIndices indices_;
  VkDevice device_;
  VkQueue graphics_queue_;
  VkQueue present_queue_;
  VkSwapchainKHR swap_chain_;
  std::vector<VkImage> swapchain_images_;
  VkFormat format_;
  VkExtent2D swapchain_extent_;
  std::vector<VkImageView> swapchain_image_views_;
  VkRenderPass render_pass_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;
  std::vector<VkFramebuffer> swapchain_frame_buffers_;

  VkCommandPool command_pool_;
  VkCommandBuffer command_buffer_;
  VkSemaphore image_available_semaphore_;
  VkSemaphore render_finished_semaphore_;
  VkFence in_flight_fence_;
};
} // namespace cg
