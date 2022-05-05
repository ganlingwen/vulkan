#include "computer_graphics_application.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "shader/shader_utils.h"

namespace cg {
namespace {
constexpr uint32_t kWidth = 800;
constexpr uint32_t kHeight = 600;

GLFWwindow *initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  return glfwCreateWindow(kWidth, kHeight, "Vulkan", nullptr, nullptr);
}
} // namespace

namespace {
VkApplicationInfo getApplicationInfo() {
  VkApplicationInfo info{};
  info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  info.pApplicationName = "Computer Graphics";
  info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  info.pEngineName = "SweetHome Engine";
  info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  info.apiVersion = VK_API_VERSION_1_0;
  return info;
}

VkInstanceCreateInfo
getInstanceCreateInfo(VkApplicationInfo *application_info) {
  VkInstanceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  info.pApplicationInfo = application_info;
  uint32_t extension_count = 0;
  const char **extensions = glfwGetRequiredInstanceExtensions(&extension_count);
  info.enabledExtensionCount = extension_count;
  info.ppEnabledExtensionNames = extensions;
  info.enabledLayerCount = 0;
  return info;
}

VkInstance initVulkan() {
  VkApplicationInfo application_info = getApplicationInfo();
  VkInstanceCreateInfo create_info = getInstanceCreateInfo(&application_info);
  VkInstance instance;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }
  return instance;
}

VkSurfaceKHR createSurface(const VkInstance &instance, GLFWwindow *window) {
  VkSurfaceKHR surface;
  if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
  return surface;
}
} // namespace

namespace {
const std::vector<const char *> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> modes;
};

QueueFamilyIndices findQueueFamilyIndices(const VkPhysicalDevice &device,
                                          const VkSurfaceKHR &surface) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count,
                                           queue_families.data());

  QueueFamilyIndices indices;
  for (int i = 0; i < count; ++i) {
    const auto &queue_family = queue_families[i];
    if (!(queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      continue;
    }
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

bool checkDeviceExtensionSupport(const VkPhysicalDevice &device) {
  std::set<std::string> required_extensions(kDeviceExtensions.begin(),
                                            kDeviceExtensions.end());
  uint32_t extension_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       nullptr);
  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       available_extensions.data());
  for (const auto &extension : available_extensions) {
    required_extensions.erase(std::string(extension.extensionName));
  }
  return required_extensions.empty();
}

SwapchainSupportDetails querySwapchainSupport(const VkPhysicalDevice &device,
                                              const VkSurfaceKHR &surface) {
  SwapchainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                            &details.capabilities);
  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count,
                                         details.formats.data());
  }
  uint32_t mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count,
                                            nullptr);
  if (mode_count != 0) {
    details.modes.resize(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count,
                                              details.modes.data());
  }
  return details;
}

bool isDeviceSuitable(const VkPhysicalDevice &device,
                      const VkSurfaceKHR &surface) {
  const QueueFamilyIndices indices = findQueueFamilyIndices(device, surface);
  if (!indices.isComplete()) {
    return false;
  }
  if (!checkDeviceExtensionSupport(device)) {
    return false;
  }
  const SwapchainSupportDetails details =
      querySwapchainSupport(device, surface);
  return !details.formats.empty() && !details.modes.empty();
}

PhysicalDevice pickPhysicalDevice(const VkInstance &instance,
                                  const VkSurfaceKHR &surface) {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  if (device_count == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

  for (const auto &device : devices) {
    if (!isDeviceSuitable(device, surface)) {
      continue;
    }
    return {
        .device = device,
        .indices = findQueueFamilyIndices(device, surface),
    };
  }
  throw std::runtime_error("failed to find a suitable GPU!");
}

std::vector<VkDeviceQueueCreateInfo>
getQueueCreateInfos(const std::set<uint32_t> &unique_queue_families,
                    float *priority) {
  std::vector<VkDeviceQueueCreateInfo> infos;
  for (uint32_t queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    info.queueFamilyIndex = queue_family;
    info.queueCount = 1;
    info.pQueuePriorities = priority;
    infos.emplace_back(std::move(info));
  }
  return infos;
}

LogicalDevice createLogicalDevice(const PhysicalDevice &physical) {
  const uint32_t graphics_family = physical.indices.graphics_family.value();
  const uint32_t present_family = physical.indices.present_family.value();
  const std::set<uint32_t> unique_queue_families = {graphics_family,
                                                    present_family};
  float queue_priority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos =
      getQueueCreateInfos(unique_queue_families, &queue_priority);
  VkPhysicalDeviceFeatures device_features{};

  VkDeviceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  info.pQueueCreateInfos = queue_create_infos.data();
  info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  info.pEnabledFeatures = &device_features;
  info.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
  info.ppEnabledExtensionNames = kDeviceExtensions.data();
  info.enabledLayerCount = 0;

  VkDevice device;
  if (vkCreateDevice(physical.device, &info, nullptr, &device) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }
  VkQueue graphics, present;
  vkGetDeviceQueue(device, graphics_family, 0, &graphics);
  vkGetDeviceQueue(device, present_family, 0, &present);

  return {.device = device, .graphics = graphics, .present = present};
}
} // namespace

namespace {
VkSurfaceFormatKHR
chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) {
  for (const auto &format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return formats[0];
}

VkPresentModeKHR
chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &modes) {
  for (const auto &mode : modes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return mode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  }
  VkExtent2D actual = {kWidth, kHeight};
  actual.width = std::clamp(actual.width, capabilities.minImageExtent.width,
                            capabilities.maxImageExtent.width);
  actual.height = std::clamp(actual.height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height);
  return actual;
}

std::vector<VkImageView> createImageViews(const std::vector<VkImage> &images,
                                          const VkFormat &format,
                                          const VkDevice &device) {
  const int num_images = images.size();
  std::vector<VkImageView> views(num_images);
  for (int i = 0; i < num_images; ++i) {
    VkImageViewCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = images[i];
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &info, nullptr, &views[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image views!");
    }
  }
  return views;
}

std::vector<VkFramebuffer>
createFramebuffers(const std::vector<VkImageView> &views,
                   const VkExtent2D &extent, const VkDevice &device) {
  std::vector<VkFramebuffer> buffers(views.size());
  for (size_t i = 0; i < views.size(); ++i) {
    VkImageView attachments[] = {views[i]};
    VkFramebufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = attachments;
    info.width = extent.width;
    info.height = extent.height;
    info.layers = 1;

    if (vkCreateFramebuffer(device, &info, nullptr, &buffers[i]) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
  return buffers;
}

Swapchain createSwapchain(const VkPhysicalDevice &physical_device,
                          const VkSurfaceKHR &surface,
                          const QueueFamilyIndices &indices,
                          const VkDevice &device) {
  const auto details = querySwapchainSupport(physical_device, surface);
  const auto surface_format = chooseSwapSurfaceFormat(details.formats);
  const auto extent = chooseSwapExtent(details.capabilities);

  uint32_t image_count = details.capabilities.minImageCount + 1;
  const uint32_t max_count = details.capabilities.maxImageCount;
  if (max_count > 0 && image_count > max_count) {
    image_count = max_count;
  }

  VkSwapchainCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  info.surface = surface;
  info.minImageCount = image_count;
  info.imageFormat = surface_format.format;
  info.imageColorSpace = surface_format.colorSpace;
  info.imageExtent = extent;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queue_family_indices[] = {indices.graphics_family.value(),
                                     indices.present_family.value()};
  if (indices.graphics_family != indices.present_family) {
    info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount = 2;
    info.pQueueFamilyIndices = queue_family_indices;
  } else {
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.queueFamilyIndexCount = 0;
    info.pQueueFamilyIndices = nullptr;
  }

  info.preTransform = details.capabilities.currentTransform;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  info.presentMode = chooseSwapPresentMode(details.modes);
  info.clipped = VK_TRUE;
  info.oldSwapchain = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain;
  if (vkCreateSwapchainKHR(device, &info, nullptr, &swapchain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain");
  }

  vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
  std::vector<VkImage> images(image_count);
  vkGetSwapchainImagesKHR(device, swapchain, &image_count, images.data());
  auto views = createImageViews(images, surface_format.format, device);
  auto buffers = createFramebuffers(views, extent, device);

  return {
      .chain = swapchain,
      .format = surface_format.format,
      .extent = extent,
      .images = std::move(images),
      .views = std::move(views),
      .buffers = std::move(buffers),
  };
}
} // namespace

namespace {
VkAttachmentDescription getAttachmentDescription(const VkFormat &format) {
  VkAttachmentDescription description{};
  description.format = format;
  description.samples = VK_SAMPLE_COUNT_1_BIT;
  description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  return description;
}

VkAttachmentReference getAttachmentReference() {
  VkAttachmentReference reference{};
  reference.attachment = 0;
  reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  return reference;
}

VkSubpassDescription getSubpassDescription(VkAttachmentReference *reference) {
  VkSubpassDescription description{};
  description.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  description.colorAttachmentCount = 1;
  description.pColorAttachments = reference;
  return description;
}

VkSubpassDependency getSubpassDependency() {
  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  return dependency;
}

VkRenderPass createRenderPass(const VkFormat &format, const VkDevice &device) {
  VkRenderPassCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

  VkAttachmentDescription attachment = getAttachmentDescription(format);
  info.attachmentCount = 1;
  info.pAttachments = &attachment;

  VkAttachmentReference reference = getAttachmentReference();
  VkSubpassDescription subpass = getSubpassDescription(&reference);
  info.subpassCount = 1;
  info.pSubpasses = &subpass;

  VkSubpassDependency dependency = getSubpassDependency();
  info.dependencyCount = 1;
  info.pDependencies = &dependency;

  VkRenderPass pass;
  if (vkCreateRenderPass(device, &info, nullptr, &pass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
  return pass;
}

VkPipelineLayout createPipelineLayout(const VkDevice &device) {
  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.setLayoutCount = 0;
  info.pSetLayouts = nullptr;
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges = nullptr;
  VkPipelineLayout layout;
  if (vkCreatePipelineLayout(device, &info, nullptr, &layout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }
  return layout;
}

std::vector<VkPipelineShaderStageCreateInfo>
getPipelineShaderStageCreateInfos(const VkShaderModule &vertex,
                                  const VkShaderModule &fragment) {
  std::vector<VkPipelineShaderStageCreateInfo> infos;
  {
    VkPipelineShaderStageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    info.module = vertex;
    info.pName = "main";
    infos.emplace_back(info);
  }
  {
    VkPipelineShaderStageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    info.module = fragment;
    info.pName = "main";
    infos.emplace_back(info);
  }
  return infos;
}

VkPipelineVertexInputStateCreateInfo getPipelineVertexInputStateCreateInfo() {
  VkPipelineVertexInputStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  info.vertexBindingDescriptionCount = 0;
  info.pVertexBindingDescriptions = nullptr;
  info.vertexAttributeDescriptionCount = 0;
  info.pVertexAttributeDescriptions = nullptr;
  return info;
}

VkPipelineInputAssemblyStateCreateInfo
getPipelineInputAssemblyStateCreateInfo() {
  VkPipelineInputAssemblyStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  info.primitiveRestartEnable = VK_FALSE;
  return info;
}

VkViewport getViewport(const VkExtent2D &extent) {
  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)extent.width;
  viewport.height = (float)extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  return viewport;
}

VkRect2D getScissor(const VkExtent2D &extent) {
  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = extent;
  return scissor;
}

VkPipelineViewportStateCreateInfo getViewportState(VkViewport *viewport,
                                                   VkRect2D *scissor) {
  VkPipelineViewportStateCreateInfo state{};
  state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  state.viewportCount = 1;
  state.pViewports = viewport;
  state.scissorCount = 1;
  state.pScissors = scissor;
  return state;
}

VkPipelineRasterizationStateCreateInfo getRasterizer() {
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f;
  rasterizer.depthBiasClamp = 0.0f;
  rasterizer.depthBiasSlopeFactor = 0.0f;
  return rasterizer;
}

VkPipelineMultisampleStateCreateInfo getMultisampling() {
  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;
  return multisampling;
}

VkPipelineColorBlendAttachmentState getColorBlendAttachment() {
  VkPipelineColorBlendAttachmentState blend{};
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blend.blendEnable = VK_FALSE;
  blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  blend.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  blend.colorBlendOp = VK_BLEND_OP_ADD;
  blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  blend.alphaBlendOp = VK_BLEND_OP_ADD;
  return blend;
}

VkPipelineColorBlendStateCreateInfo
getColorBlend(VkPipelineColorBlendAttachmentState *attachment) {
  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = attachment;
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;
  return color_blending;
}

std::vector<VkDynamicState> getDynamicStates() {
  return {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_LINE_WIDTH};
}

VkPipelineDynamicStateCreateInfo
getDynamicStateCreateInfo(std::vector<VkDynamicState> &states) {
  VkPipelineDynamicStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  info.dynamicStateCount = states.size();
  info.pDynamicStates = states.data();
  return info;
}

VkPipeline createGraphicsPipeline(const VkDevice &device,
                                  const VkExtent2D &extent,
                                  const VkPipelineLayout &layout,
                                  const VkRenderPass &pass) {
  VkGraphicsPipelineCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

  const VkShaderModule vert_shader = createShaderModule("shader.vert", device);
  const VkShaderModule frag_shader = createShaderModule("shader.frag", device);
  auto shader_stages =
      getPipelineShaderStageCreateInfos(vert_shader, frag_shader);
  info.stageCount = 2;
  info.pStages = shader_stages.data();

  auto vertex_input_info = getPipelineVertexInputStateCreateInfo();
  info.pVertexInputState = &vertex_input_info;

  auto input_assembly = getPipelineInputAssemblyStateCreateInfo();
  info.pInputAssemblyState = &input_assembly;

  auto viewport = getViewport(extent);
  auto scissor = getScissor(extent);
  auto viewport_state = getViewportState(&viewport, &scissor);
  info.pViewportState = &viewport_state;

  auto rasterizer = getRasterizer();
  info.pRasterizationState = &rasterizer;

  auto multisampling = getMultisampling();
  info.pMultisampleState = &multisampling;

  info.pDepthStencilState = nullptr;

  auto color_blend_attachment = getColorBlendAttachment();
  auto color_blending = getColorBlend(&color_blend_attachment);
  info.pColorBlendState = &color_blending;

  info.pDynamicState = nullptr;
  info.layout = layout;
  info.renderPass = pass;
  info.subpass = 0;
  info.basePipelineHandle = VK_NULL_HANDLE;
  info.basePipelineIndex = -1;

  VkPipeline pipeline;
  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                &pipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }
  vkDestroyShaderModule(device, frag_shader, nullptr);
  vkDestroyShaderModule(device, vert_shader, nullptr);
  return pipeline;
}
} // namespace

namespace {
VkCommandPool createCommandPool(const QueueFamilyIndices &indices,
                                const VkDevice &device) {
  VkCommandPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  info.queueFamilyIndex = indices.graphics_family.value();
  VkCommandPool pool;
  if (vkCreateCommandPool(device, &info, nullptr, &pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
  return pool;
}

VkCommandBuffer createCommandBuffer(const VkCommandPool &pool,
                                    const VkDevice &device) {
  VkCommandBufferAllocateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.commandPool = pool;
  info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  info.commandBufferCount = 1;
  VkCommandBuffer buffer;
  if (vkAllocateCommandBuffers(device, &info, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffer!");
  }
  return buffer;
}

VkSemaphore createSemaphore(const VkDevice &device) {
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore semaphore;
  if (vkCreateSemaphore(device, &info, nullptr, &semaphore) != VK_SUCCESS) {
    throw std::runtime_error("failed to create semaphores!");
  }
  return semaphore;
}

VkFence createFence(const VkDevice &device) {
  VkFenceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFence fence;
  if (vkCreateFence(device, &info, nullptr, &fence) != VK_SUCCESS) {
    throw std::runtime_error("failed to create semaphores!");
  }
  return fence;
}
} // namespace

ComputerGraphicsApplication::ComputerGraphicsApplication() {
  window_ = initWindow();
  instance_ = initVulkan();
  surface_ = createSurface(instance_, window_);

  physical_ = pickPhysicalDevice(instance_, surface_);
  logical_ = createLogicalDevice(physical_);
  swapchain_ = createSwapchain(physical_.device, surface_, physical_.indices,
                               logical_.device);

  render_pass_ = createRenderPass(swapchain_.format, logical_.device);
  pipeline_layout_ = createPipelineLayout(logical_.device);
  graphics_pipeline_ = createGraphicsPipeline(
      logical_.device, swapchain_.extent, pipeline_layout_, render_pass_);
  command_pool_ = createCommandPool(physical_.indices, logical_.device);
  command_buffer_ = createCommandBuffer(command_pool_, logical_.device);

  image_available_semaphore_ = createSemaphore(logical_.device);
  render_finished_semaphore_ = createSemaphore(logical_.device);
  in_flight_fence_ = createFence(logical_.device);
}

ComputerGraphicsApplication::~ComputerGraphicsApplication() {
  vkDestroySemaphore(logical_.device, image_available_semaphore_, nullptr);
  vkDestroySemaphore(logical_.device, render_finished_semaphore_, nullptr);
  vkDestroyFence(logical_.device, in_flight_fence_, nullptr);

  vkDestroyCommandPool(logical_.device, command_pool_, nullptr);
  vkDestroyPipeline(logical_.device, graphics_pipeline_, nullptr);
  vkDestroyPipelineLayout(logical_.device, pipeline_layout_, nullptr);
  vkDestroyRenderPass(logical_.device, render_pass_, nullptr);

  swapchain_.destroy(logical_.device);

  vkDestroyDevice(logical_.device, nullptr);
  vkDestroySurfaceKHR(instance_, surface_, nullptr);
  vkDestroyInstance(instance_, nullptr);
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void ComputerGraphicsApplication::run() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    drawFrame();
  }
  vkDeviceWaitIdle(logical_.device);
}

void ComputerGraphicsApplication::drawFrame() {
  vkWaitForFences(logical_.device, 1, &in_flight_fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(logical_.device, 1, &in_flight_fence_);

  uint32_t image_index;
  vkAcquireNextImageKHR(logical_.device, swapchain_.chain, UINT64_MAX,
                        image_available_semaphore_, VK_NULL_HANDLE,
                        &image_index);

  vkResetCommandBuffer(command_buffer_, 0);
  recordCommandBuffer(command_buffer_, image_index);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  VkSemaphore wait_semaphores[] = {image_available_semaphore_};
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer_;
  VkSemaphore signal_semaphores[] = {render_finished_semaphore_};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;
  if (vkQueueSubmit(logical_.graphics, 1, &submit_info, in_flight_fence_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;
  VkSwapchainKHR swapchains[] = {swapchain_.chain};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &image_index;
  present_info.pResults = nullptr;
  vkQueuePresentKHR(logical_.present, &present_info);
}

void ComputerGraphicsApplication::recordCommandBuffer(
    VkCommandBuffer command_buffer, uint32_t image_index) {
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = 0;
  begin_info.pInheritanceInfo = nullptr;
  if (vkBeginCommandBuffer(command_buffer_, &begin_info) != VK_SUCCESS) {
    throw std::runtime_error("failed to begin recording command buffer!");
  }

  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = render_pass_;
  render_pass_info.framebuffer = swapchain_.buffers[image_index];
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = swapchain_.extent;
  VkClearValue clear_value = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
  render_pass_info.clearValueCount = 1;
  render_pass_info.pClearValues = &clear_value;

  vkCmdBeginRenderPass(command_buffer_, &render_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphics_pipeline_);
  vkCmdDraw(command_buffer_, 3, 1, 0, 0);
  vkCmdEndRenderPass(command_buffer_);
  if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) {
    throw std::runtime_error("failed to record command buffer!");
  }
}
} // namespace cg
