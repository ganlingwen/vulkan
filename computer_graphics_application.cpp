#include "computer_graphics_application.h"

#include <algorithm>
#include <string>
#include <set>
#include <stdexcept>
#include <vector>

namespace cg {
namespace {
constexpr uint32_t kWidth = 800;
constexpr uint32_t kHeight = 600;

const std::vector<const char*> device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> modes;
};

QueueFamilyIndices findQueueFamilyIndices(
    VkPhysicalDevice device, VkSurfaceKHR surface) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queue_families.data());

  QueueFamilyIndices indices;
  for (int i = 0; i < count; ++i) {
    const auto& queue_family = queue_families[i];
    if (!(queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
    indices.graphics_family = i;

    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
    if (present_support) {
      indices.present_family = i;
      return indices;
    }
  }
  return indices;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
  std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

  uint32_t extension_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

  for (const auto& extension : available_extensions) {
    required_extensions.erase(std::string(extension.extensionName));
  }
  return required_extensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(
    VkPhysicalDevice device, VkSurfaceKHR surface) {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
  }

  uint32_t mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, nullptr);
  if (mode_count != 0) {
    details.modes.resize(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, details.modes.data());
  }
  return details;
}

bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
  QueueFamilyIndices indices = findQueueFamilyIndices(device, surface);
  if (!indices.isComplete()) return false;
  if (!checkDeviceExtensionSupport(device)) return false;
  const SwapChainSupportDetails details = querySwapChainSupport(device, surface);
  return !details.formats.empty() && !details.modes.empty();
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& formats) {
  for (const auto& format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return formats[0];
}

VkPresentModeKHR chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& modes) {
  for (const auto& mode : modes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  }
  VkExtent2D actual = {kWidth, kHeight};
  actual.width = std::clamp(actual.width,
      capabilities.minImageExtent.width,
      capabilities.maxImageExtent.width);
  actual.height = std::clamp(actual.height,
      capabilities.minImageExtent.height,
      capabilities.maxImageExtent.height);
  return actual;
}
} // namespace

ComputerGraphicsApplication::ComputerGraphicsApplication() {
  initWindow();
  initVulkan();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createGraphicsPipeline();
}

ComputerGraphicsApplication::~ComputerGraphicsApplication() {
  for (auto image_view : swapchain_image_views_) {
    vkDestroyImageView(device_, image_view, nullptr);
  }
  vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
  vkDestroyDevice(device_, nullptr);
  vkDestroySurfaceKHR(instance_, surface_, nullptr);
  vkDestroyInstance(instance_, nullptr);
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void ComputerGraphicsApplication::run() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
  }
}

void ComputerGraphicsApplication::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  window_ = glfwCreateWindow(kWidth, kHeight, "Vulkan", nullptr, nullptr);
}

void ComputerGraphicsApplication::initVulkan() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Computer Graphics";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "SweetHome Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  uint32_t extension_count = 0;
  const char** extensions = glfwGetRequiredInstanceExtensions(&extension_count);
  create_info.enabledExtensionCount = extension_count;
  create_info.ppEnabledExtensionNames = extensions;
  create_info.enabledLayerCount = 0;

  if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }
}

void ComputerGraphicsApplication::createSurface() {
  if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
}

void ComputerGraphicsApplication::pickPhysicalDevice() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  if (device_count == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  for (const auto& device : devices) {
    if (!isDeviceSuitable(device, surface_)) continue;
    indices_ = findQueueFamilyIndices(device, surface_);
    physical_device_ = device;
    return;
  }
  throw std::runtime_error("failed to find a suitable GPU!");
}

void ComputerGraphicsApplication::createLogicalDevice() {
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families = {
    indices_.graphics_family.value(), indices_.present_family.value()};
  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.emplace_back(queue_create_info);
  }

  VkPhysicalDeviceFeatures device_features{};

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
  create_info.ppEnabledExtensionNames = device_extensions.data();
  create_info.enabledLayerCount = 0;
  if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device_, indices_.graphics_family.value(), 0, &graphics_queue_);
  vkGetDeviceQueue(device_, indices_.present_family.value(), 0, &present_queue_);
}

void ComputerGraphicsApplication::createSwapChain() {
  const auto details = querySwapChainSupport(physical_device_, surface_);
  const VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(details.formats);
  const VkPresentModeKHR present_mode = chooseSwapPresentMode(details.modes);
  const VkExtent2D extent = chooseSwapExtent(details.capabilities);

  uint32_t image_count = details.capabilities.minImageCount + 1;
  const uint32_t max_count = details.capabilities.maxImageCount;
  if (max_count > 0 && image_count > max_count) image_count = max_count;

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface_;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queue_family_indices[] = {
    indices_.graphics_family.value(), indices_.present_family.value()};
  if (indices_.graphics_family != indices_.present_family) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
  }

  create_info.preTransform = details.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain");
  }

  vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
  swapchain_images_.resize(image_count);
  vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, swapchain_images_.data());
  format_ = surface_format.format;
  swapchain_extent_ = extent;
}

void ComputerGraphicsApplication::createImageViews() {
  const int num_images = swapchain_images_.size();
  swapchain_image_views_.resize(num_images);
  for (int i = 0; i < num_images; ++i) {
    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = swapchain_images_[i];
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = format_;
    create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device_, &create_info, nullptr, &swapchain_image_views_[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image views!");
    }
  }
}

void ComputerGraphicsApplication::createGraphicsPipeline() {

}
} // namespace cg
