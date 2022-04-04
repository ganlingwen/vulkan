#include "computer_graphics_application.h"

#include <set>
#include <stdexcept>
#include <vector>

namespace cg {
namespace {
constexpr uint32_t kWidth = 800;
constexpr uint32_t kHeight = 600;

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
} // namespace

ComputerGraphicsApplication::ComputerGraphicsApplication() {
  initWindow();
  initVulkan();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
}

ComputerGraphicsApplication::~ComputerGraphicsApplication() {
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
    indices_ = findQueueFamilyIndices(device, surface_);
    if (indices_.isCompute()) {
      physical_device_ = device;
      return;
    }
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
  create_info.enabledExtensionCount = 0;
  create_info.enabledLayerCount = 0;
  if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device_, indices_.graphics_family.value(), 0, &graphics_queue_);
  vkGetDeviceQueue(device_, indices_.present_family.value(), 0, &present_queue_);
}
} // namespace cg
