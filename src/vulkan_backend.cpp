// SPDX-License-Identifier: GPL-3.0-or-later
//
// vulkan_backend.cpp — Vulkan compute FFT backend.
//
// This is necessarily verbose because Vulkan IS verbose. We've structured
// it so the FFT algorithm itself (~50 lines) is clearly separated from
// the resource management (~500 lines).
//
// Resource lifecycle (in construction order, destroyed in REVERSE):
//   1. VkInstance
//   2. VkPhysicalDevice (chosen)
//   3. VkDevice + VkQueue
//   4. VkCommandPool
//   5. VkDescriptorSetLayout (one per shader)
//   6. VkPipelineLayout (one per shader)
//   7. VkShaderModule + VkPipeline (one per shader)
//   8. VkDescriptorPool (sized for our cached plans)
//   9. Per-(fft_size) cached plans:
//      - 2x VkBuffer + VkDeviceMemory (ping-pong)
//      - 1x VkBuffer + VkDeviceMemory (host-coherent staging for upload/download)
//      - 1x VkDescriptorSet for fft_stage (we re-bind on each call)
//
// Failure handling:
//   * Any VkResult != VK_SUCCESS during construction -> throw runtime_error
//   * Failures during fft_* calls leave the backend in an unspecified state.
//     Callers should treat that backend as failed and use a different one
//     (the hybrid backend does this automatically).
#include "aboba/vulkan_backend.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "vulkan_shaders_gen/fft_stage.h"
#include "vulkan_shaders_gen/r2c_pack.h"
#include "vulkan_shaders_gen/c2r_finish.h"
#include "vulkan_shaders_gen/bitrev_permute.h"

namespace aboba {

namespace {

#define VK_CHECK(expr)                                                  \
    do {                                                                \
        VkResult _r = (expr);                                           \
        if (_r != VK_SUCCESS) {                                         \
            throw std::runtime_error(std::string(#expr) + " failed: "   \
                + std::to_string(static_cast<int>(_r)));                \
        }                                                               \
    } while (0)

inline bool is_power_of_two(std::size_t n) noexcept {
    return n >= 2 && (n & (n - 1)) == 0;
}

inline std::size_t log2_size(std::size_t n) noexcept {
    std::size_t r = 0;
    while ((std::size_t{1} << r) < n) ++r;
    return r;
}

}  // namespace

// ============================================================
// Impl
// ============================================================
class VulkanBackendImpl {
public:
    VulkanBackendImpl(bool validate, bool prefer_discrete);
    ~VulkanBackendImpl();

    VulkanBackendImpl(const VulkanBackendImpl&)            = delete;
    VulkanBackendImpl& operator=(const VulkanBackendImpl&) = delete;

    void fft_r2c(const float* in, std::complex<float>* out, std::size_t n);
    void fft_c2r(const std::complex<float>* in, float* out, std::size_t n);

    const std::string& device_name() const noexcept { return device_name_; }
    const std::string& driver_info() const noexcept { return driver_info_; }
    bool is_software() const noexcept { return is_software_; }

private:
    // ---- Per-(fft_size) cached plan ----
    struct Plan {
        std::size_t n           = 0;
        // Two device-local buffers for ping-pong compute
        VkBuffer       buf_a    = VK_NULL_HANDLE;
        VkBuffer       buf_b    = VK_NULL_HANDLE;
        VkDeviceMemory mem_a    = VK_NULL_HANDLE;
        VkDeviceMemory mem_b    = VK_NULL_HANDLE;
        // Real-side buffer (for r2c input / c2r output)
        VkBuffer       buf_real = VK_NULL_HANDLE;
        VkDeviceMemory mem_real = VK_NULL_HANDLE;
        // Host-visible staging for both upload and download
        VkBuffer       buf_stage  = VK_NULL_HANDLE;
        VkDeviceMemory mem_stage  = VK_NULL_HANDLE;
        VkDeviceSize   stage_size = 0;
        void*          stage_mapped = nullptr;
    };

    Plan& get_or_create_plan(std::size_t n);

    void run_fft(Plan& plan, int direction);

    void create_instance(bool validate);
    void pick_physical_device(bool prefer_discrete);
    void create_device();
    void create_pipelines();
    void create_descriptor_pool();
    void create_command_pool();

    VkShaderModule make_shader_module(const std::uint32_t* spv,
                                      std::size_t size_bytes);
    void destroy_pipeline_stuff();

    // Find memory type matching memoryTypeBits and required properties
    std::uint32_t find_memory_type(std::uint32_t type_bits,
                                   VkMemoryPropertyFlags props);

    void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props,
                       VkBuffer& buffer, VkDeviceMemory& memory);

    // Record + submit a pipeline dispatch on the current command buffer.
    // `groups` is the number of workgroups to dispatch.
    void dispatch_one(VkPipeline pipeline, VkPipelineLayout layout,
                      VkDescriptorSet set, void* push, std::uint32_t push_size,
                      std::uint32_t groups);

    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_    = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          queue_          = VK_NULL_HANDLE;
    std::uint32_t    queue_family_   = 0;
    VkCommandPool    cmd_pool_       = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf_        = VK_NULL_HANDLE;
    VkFence          fence_          = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_      = VK_NULL_HANDLE;

    // Pipelines: bitrev_permute (2-buf), r2c_pack (2-buf), c2r_finish (2-buf),
    //            fft_stage (1-buf, in-place)
    VkShaderModule        sh_fft_      = VK_NULL_HANDLE;
    VkShaderModule        sh_r2c_      = VK_NULL_HANDLE;
    VkShaderModule        sh_c2r_      = VK_NULL_HANDLE;
    VkShaderModule        sh_bitrev_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_one_buf_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_two_buf_ = VK_NULL_HANDLE;
    VkPipelineLayout      pl_fft_      = VK_NULL_HANDLE;
    VkPipelineLayout      pl_r2c_      = VK_NULL_HANDLE;
    VkPipelineLayout      pl_c2r_      = VK_NULL_HANDLE;
    VkPipelineLayout      pl_bitrev_   = VK_NULL_HANDLE;
    VkPipeline            pipe_fft_    = VK_NULL_HANDLE;
    VkPipeline            pipe_r2c_    = VK_NULL_HANDLE;
    VkPipeline            pipe_c2r_    = VK_NULL_HANDLE;
    VkPipeline            pipe_bitrev_ = VK_NULL_HANDLE;

    VkPhysicalDeviceMemoryProperties mem_props_{};
    std::string device_name_;
    std::string driver_info_;
    bool is_software_ = false;

    std::unordered_map<std::size_t, Plan> plans_;
    std::mutex mtx_;
};

// ============================================================
// Construction
// ============================================================
VulkanBackendImpl::VulkanBackendImpl(bool validate, bool prefer_discrete) {
    create_instance(validate);
    pick_physical_device(prefer_discrete);
    create_device();
    create_command_pool();
    create_descriptor_pool();
    create_pipelines();
}

VulkanBackendImpl::~VulkanBackendImpl() {
    // Wait for any in-flight work
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }

    // Destroy cached plans
    for (auto& [n, plan] : plans_) {
        if (plan.stage_mapped) vkUnmapMemory(device_, plan.mem_stage);
        if (plan.buf_stage) vkDestroyBuffer(device_, plan.buf_stage, nullptr);
        if (plan.mem_stage) vkFreeMemory(device_, plan.mem_stage, nullptr);
        if (plan.buf_real)  vkDestroyBuffer(device_, plan.buf_real, nullptr);
        if (plan.mem_real)  vkFreeMemory(device_, plan.mem_real, nullptr);
        if (plan.buf_a)     vkDestroyBuffer(device_, plan.buf_a, nullptr);
        if (plan.mem_a)     vkFreeMemory(device_, plan.mem_a, nullptr);
        if (plan.buf_b)     vkDestroyBuffer(device_, plan.buf_b, nullptr);
        if (plan.mem_b)     vkFreeMemory(device_, plan.mem_b, nullptr);
    }
    plans_.clear();

    destroy_pipeline_stuff();

    if (desc_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    if (fence_ != VK_NULL_HANDLE)
        vkDestroyFence(device_, fence_, nullptr);
    if (cmd_pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

void VulkanBackendImpl::destroy_pipeline_stuff() {
    if (pipe_fft_)    vkDestroyPipeline(device_, pipe_fft_, nullptr);
    if (pipe_r2c_)    vkDestroyPipeline(device_, pipe_r2c_, nullptr);
    if (pipe_c2r_)    vkDestroyPipeline(device_, pipe_c2r_, nullptr);
    if (pipe_bitrev_) vkDestroyPipeline(device_, pipe_bitrev_, nullptr);
    if (pl_fft_)      vkDestroyPipelineLayout(device_, pl_fft_, nullptr);
    if (pl_r2c_)      vkDestroyPipelineLayout(device_, pl_r2c_, nullptr);
    if (pl_c2r_)      vkDestroyPipelineLayout(device_, pl_c2r_, nullptr);
    if (pl_bitrev_)   vkDestroyPipelineLayout(device_, pl_bitrev_, nullptr);
    if (dsl_one_buf_) vkDestroyDescriptorSetLayout(device_, dsl_one_buf_, nullptr);
    if (dsl_two_buf_) vkDestroyDescriptorSetLayout(device_, dsl_two_buf_, nullptr);
    if (sh_fft_)      vkDestroyShaderModule(device_, sh_fft_, nullptr);
    if (sh_r2c_)      vkDestroyShaderModule(device_, sh_r2c_, nullptr);
    if (sh_c2r_)      vkDestroyShaderModule(device_, sh_c2r_, nullptr);
    if (sh_bitrev_)   vkDestroyShaderModule(device_, sh_bitrev_, nullptr);
}

void VulkanBackendImpl::create_instance(bool validate) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Aboba Vocoder";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "Aboba";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    std::vector<const char*> layers;
    if (validate) {
        const char* env = std::getenv("ABOBA_VULKAN_VALIDATE");
        if (env || validate) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }
    ci.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();

    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
}

void VulkanBackendImpl::pick_physical_device(bool prefer_discrete) {
    std::uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    if (count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devs(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devs.data()));

    // Score: prefer discrete (4), then integrated (3), then virtual (2),
    // then CPU/software (1).
    VkPhysicalDevice best = VK_NULL_HANDLE;
    int best_score = -1;
    VkPhysicalDeviceProperties best_props{};
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        int s = 0;
        switch (p.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   s = prefer_discrete ? 4 : 3; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: s = prefer_discrete ? 3 : 4; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    s = 2; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            s = 1; break;
            default: s = 0;
        }
        // Need compute queue
        std::uint32_t qfc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfc, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qfc);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfc, qfp.data());
        bool has_compute = false;
        for (auto& q : qfp) {
            if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) { has_compute = true; break; }
        }
        if (!has_compute) continue;

        if (s > best_score) {
            best_score = s;
            best = d;
            best_props = p;
        }
    }
    if (best == VK_NULL_HANDLE) {
        throw std::runtime_error("No Vulkan device with a compute queue");
    }
    phys_device_ = best;
    device_name_ = best_props.deviceName;
    is_software_ = (best_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU);

    // Memory props
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mem_props_);

    // Driver info (extension VK_KHR_driver_properties may or may not be there;
    // try the cleaner path if available).
    char drv[128];
    std::snprintf(drv, sizeof(drv), "API %u.%u.%u driver 0x%x",
                  VK_VERSION_MAJOR(best_props.apiVersion),
                  VK_VERSION_MINOR(best_props.apiVersion),
                  VK_VERSION_PATCH(best_props.apiVersion),
                  best_props.driverVersion);
    driver_info_ = drv;

    // Pick a compute queue family
    std::uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qfc, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qfc, qfp.data());
    queue_family_ = 0;
    for (std::uint32_t i = 0; i < qfc; ++i) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_ = i;
            break;
        }
    }
}

void VulkanBackendImpl::create_device() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(phys_device_, &dci, nullptr, &device_));
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

void VulkanBackendImpl::create_command_pool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = queue_family_;
    VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmd_pool_));

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = cmd_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd_buf_));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device_, &fci, nullptr, &fence_));
}

void VulkanBackendImpl::create_descriptor_pool() {
    // Worst case: one descriptor set per shader per plan size we cache.
    // We cap cached plans at ~16, so 16*3 sets max; each set has 2
    // storage buffer descriptors.
    VkDescriptorPoolSize size{};
    size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 256;

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = 128;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &size;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &desc_pool_));
}

VkShaderModule VulkanBackendImpl::make_shader_module(const std::uint32_t* spv,
                                                     std::size_t size_bytes) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size_bytes;
    ci.pCode = spv;
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &m));
    return m;
}

void VulkanBackendImpl::create_pipelines() {
    using namespace vk_shaders;
    sh_fft_    = make_shader_module(fft_stage_spv,      fft_stage_spv_size_bytes);
    sh_r2c_    = make_shader_module(r2c_pack_spv,       r2c_pack_spv_size_bytes);
    sh_c2r_    = make_shader_module(c2r_finish_spv,     c2r_finish_spv_size_bytes);
    sh_bitrev_ = make_shader_module(bitrev_permute_spv, bitrev_permute_spv_size_bytes);

    // ---- One-buffer DSL (for in-place fft_stage) ----
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &dsl_one_buf_));
    }

    // ---- Two-buffer DSL (for r2c, c2r, bitrev: separate src/dst) ----
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (int i = 0; i < 2; ++i) {
            bindings[i].binding = static_cast<std::uint32_t>(i);
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 2;
        ci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &dsl_two_buf_));
    }

    auto make_layout = [&](VkDescriptorSetLayout dsl, std::uint32_t pc_bytes) {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = pc_bytes;
        VkPipelineLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &dsl;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pcr;
        VkPipelineLayout l;
        VK_CHECK(vkCreatePipelineLayout(device_, &ci, nullptr, &l));
        return l;
    };
    // fft_stage: 1 buf, 16 bytes push (n, m_half, dir, pad)
    pl_fft_    = make_layout(dsl_one_buf_, 16);
    // r2c_pack:  2 bufs, 4 bytes (n)
    pl_r2c_    = make_layout(dsl_two_buf_, 4);
    // c2r_finish: 2 bufs, 4 bytes (n)
    pl_c2r_    = make_layout(dsl_two_buf_, 4);
    // bitrev_permute: 2 bufs, 8 bytes (n, log2_n)
    pl_bitrev_ = make_layout(dsl_two_buf_, 8);

    auto make_pipeline = [&](VkShaderModule sm, VkPipelineLayout pl) {
        VkPipelineShaderStageCreateInfo ssci{};
        ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ssci.module = sm;
        ssci.pName = "main";
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = ssci;
        ci.layout = pl;
        VkPipeline p;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
        return p;
    };
    pipe_fft_    = make_pipeline(sh_fft_, pl_fft_);
    pipe_r2c_    = make_pipeline(sh_r2c_, pl_r2c_);
    pipe_c2r_    = make_pipeline(sh_c2r_, pl_c2r_);
    pipe_bitrev_ = make_pipeline(sh_bitrev_, pl_bitrev_);
}

std::uint32_t VulkanBackendImpl::find_memory_type(std::uint32_t type_bits,
                                                  VkMemoryPropertyFlags props) {
    for (std::uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mem_props_.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Vulkan: no suitable memory type");
}

void VulkanBackendImpl::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                      VkMemoryPropertyFlags props,
                                      VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device_, buffer, memory, 0));
}

VulkanBackendImpl::Plan& VulkanBackendImpl::get_or_create_plan(std::size_t n) {
    if (auto it = plans_.find(n); it != plans_.end()) return it->second;

    Plan p;
    p.n = n;

    const VkDeviceSize complex_size = sizeof(float) * 2 * n;
    const VkDeviceSize real_size    = sizeof(float) * n;
    // Stage buffer: large enough for whichever direction we're going
    p.stage_size = std::max(complex_size, real_size);

    const VkBufferUsageFlags storage_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Device-local buffers for compute
    create_buffer(complex_size, storage_usage,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  p.buf_a, p.mem_a);
    create_buffer(complex_size, storage_usage,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  p.buf_b, p.mem_b);
    create_buffer(real_size, storage_usage,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  p.buf_real, p.mem_real);

    // Host-visible staging
    create_buffer(p.stage_size,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  p.buf_stage, p.mem_stage);
    VK_CHECK(vkMapMemory(device_, p.mem_stage, 0, p.stage_size, 0, &p.stage_mapped));

    auto [it, _] = plans_.emplace(n, std::move(p));
    return it->second;
}

void VulkanBackendImpl::dispatch_one(VkPipeline pipeline,
                                     VkPipelineLayout layout,
                                     VkDescriptorSet set,
                                     void* push, std::uint32_t push_size,
                                     std::uint32_t groups) {
    vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout, 0, 1, &set, 0, nullptr);
    if (push_size > 0) {
        vkCmdPushConstants(cmd_buf_, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_size, push);
    }
    vkCmdDispatch(cmd_buf_, groups, 1, 1);

    // Storage-buffer write -> storage-buffer read barrier between stages
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd_buf_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

// ============================================================
// The actual FFT call
// ============================================================
//
// For r2c_fft we:
//   1. Upload real-valued input to staging
//   2. Copy stage -> buf_real
//   3. Dispatch r2c_pack: buf_real -> buf_a (complex)
//   4. log2(n) times: dispatch fft_stage with ping-pong between buf_a/buf_b
//   5. Final result is in buf_a or buf_b depending on parity
//   6. Copy final -> stage
//   7. vkQueueSubmit + wait for fence
//   8. Read stage into the user's output buffer (taking only first N/2+1 bins)
//
// For c2r_fft we do the reverse: upload N/2+1 complex bins, expand to full
// N (Hermitian symmetry), inverse FFT, c2r_finish.

void VulkanBackendImpl::fft_r2c(const float* input, std::complex<float>* output,
                                std::size_t n) {
    if (!is_power_of_two(n) || n < 4) {
        throw std::invalid_argument("VulkanBackend: fft size must be power of 2 >= 4");
    }
    std::lock_guard<std::mutex> lock(mtx_);
    Plan& plan = get_or_create_plan(n);

    // 1. Upload real input
    std::memcpy(plan.stage_mapped, input, n * sizeof(float));

    VK_CHECK(vkResetCommandBuffer(cmd_buf_, 0));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buf_, &bi));

    // stage -> buf_real
    {
        VkBufferCopy copy{}; copy.size = sizeof(float) * n;
        vkCmdCopyBuffer(cmd_buf_, plan.buf_stage, plan.buf_real, 1, &copy);
        VkBufferMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.buffer = plan.buf_real;
        mb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd_buf_,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &mb, 0, nullptr);
    }

    // Allocate descriptor sets:
    //   1 x two-buf (r2c_pack)
    //   1 x two-buf (bitrev_permute)
    //   log2(N) x one-buf (fft_stage in-place)
    const std::size_t stages = log2_size(n);
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(dsl_two_buf_);  // r2c_pack
    layouts.push_back(dsl_two_buf_);  // bitrev_permute
    for (std::size_t i = 0; i < stages; ++i) layouts.push_back(dsl_one_buf_);

    std::vector<VkDescriptorSet> sets(layouts.size(), VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = desc_pool_;
    dai.descriptorSetCount = static_cast<std::uint32_t>(layouts.size());
    dai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(device_, &dai, sets.data()));

    auto write_two = [&](VkDescriptorSet set, VkBuffer b0, VkBuffer b1) {
        VkDescriptorBufferInfo dbi[2]{};
        dbi[0].buffer = b0; dbi[0].offset = 0; dbi[0].range = VK_WHOLE_SIZE;
        dbi[1].buffer = b1; dbi[1].offset = 0; dbi[1].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet ws[2]{};
        for (int i = 0; i < 2; ++i) {
            ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet = set;
            ws[i].dstBinding = static_cast<std::uint32_t>(i);
            ws[i].descriptorCount = 1;
            ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(device_, 2, ws, 0, nullptr);
    };
    auto write_one = [&](VkDescriptorSet set, VkBuffer b0) {
        VkDescriptorBufferInfo dbi{};
        dbi.buffer = b0; dbi.offset = 0; dbi.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet ws{};
        ws.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws.dstSet = set;
        ws.dstBinding = 0;
        ws.descriptorCount = 1;
        ws.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(device_, 1, &ws, 0, nullptr);
    };

    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t log2_n_u = static_cast<std::uint32_t>(stages);

    // r2c_pack: buf_real -> buf_a (complex with imag=0)
    write_two(sets[0], plan.buf_real, plan.buf_a);
    dispatch_one(pipe_r2c_, pl_r2c_, sets[0], (void*)&n_u, sizeof(n_u),
                 static_cast<std::uint32_t>((n + 63) / 64));

    // bitrev_permute: buf_a -> buf_b
    {
        struct { std::uint32_t n; std::uint32_t log2_n; } push{n_u, log2_n_u};
        write_two(sets[1], plan.buf_a, plan.buf_b);
        dispatch_one(pipe_bitrev_, pl_bitrev_, sets[1], &push, sizeof(push),
                     static_cast<std::uint32_t>((n + 63) / 64));
    }

    // In-place FFT stages on buf_b
    for (std::size_t s = 0; s < stages; ++s) {
        write_one(sets[2 + s], plan.buf_b);
        struct { std::uint32_t n; std::uint32_t m_half; int dir; std::uint32_t pad; } push;
        push.n = n_u;
        push.m_half = static_cast<std::uint32_t>(1u << s);
        push.dir = +1;
        push.pad = 0;
        dispatch_one(pipe_fft_, pl_fft_, sets[2 + s], &push, sizeof(push),
                     static_cast<std::uint32_t>(((n / 2) + 63) / 64));
    }

    // Copy result to stage
    VkBufferCopy copyback{};
    copyback.size = sizeof(float) * 2 * n;
    vkCmdCopyBuffer(cmd_buf_, plan.buf_b, plan.buf_stage, 1, &copyback);

    VK_CHECK(vkEndCommandBuffer(cmd_buf_));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf_;
    VK_CHECK(vkResetFences(device_, 1, &fence_));
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
    VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));

    const std::size_t n_bins = n / 2 + 1;
    std::memcpy(output, plan.stage_mapped, n_bins * sizeof(std::complex<float>));

    vkFreeDescriptorSets(device_, desc_pool_,
                         static_cast<std::uint32_t>(sets.size()), sets.data());
}

void VulkanBackendImpl::fft_c2r(const std::complex<float>* input, float* output,
                                std::size_t n) {
    if (!is_power_of_two(n) || n < 4) {
        throw std::invalid_argument("VulkanBackend: fft size must be power of 2 >= 4");
    }
    std::lock_guard<std::mutex> lock(mtx_);
    Plan& plan = get_or_create_plan(n);

    // c2r: caller gives N/2+1 bins. Expand to full N via Hermitian symmetry.
    const std::size_t n_bins = n / 2 + 1;
    auto* stage_complex = static_cast<std::complex<float>*>(plan.stage_mapped);
    std::memcpy(stage_complex, input, n_bins * sizeof(std::complex<float>));
    for (std::size_t k = 1; k < n / 2; ++k) {
        stage_complex[n - k] = std::conj(input[k]);
    }
    stage_complex[0]     = std::complex<float>(stage_complex[0].real(),     0.0f);
    stage_complex[n / 2] = std::complex<float>(stage_complex[n / 2].real(), 0.0f);

    VK_CHECK(vkResetCommandBuffer(cmd_buf_, 0));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buf_, &bi));

    // stage -> buf_a
    {
        VkBufferCopy copy{}; copy.size = sizeof(float) * 2 * n;
        vkCmdCopyBuffer(cmd_buf_, plan.buf_stage, plan.buf_a, 1, &copy);
        VkBufferMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.buffer = plan.buf_a;
        mb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd_buf_,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &mb, 0, nullptr);
    }

    // Descriptor sets: bitrev (two-buf) + log2(N) fft_stages (one-buf) + c2r_finish (two-buf)
    const std::size_t stages = log2_size(n);
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(dsl_two_buf_);  // bitrev_permute
    for (std::size_t i = 0; i < stages; ++i) layouts.push_back(dsl_one_buf_);
    layouts.push_back(dsl_two_buf_);  // c2r_finish

    std::vector<VkDescriptorSet> sets(layouts.size(), VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = desc_pool_;
    dai.descriptorSetCount = static_cast<std::uint32_t>(layouts.size());
    dai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(device_, &dai, sets.data()));

    auto write_two = [&](VkDescriptorSet set, VkBuffer b0, VkBuffer b1) {
        VkDescriptorBufferInfo dbi[2]{};
        dbi[0].buffer = b0; dbi[0].offset = 0; dbi[0].range = VK_WHOLE_SIZE;
        dbi[1].buffer = b1; dbi[1].offset = 0; dbi[1].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet ws[2]{};
        for (int i = 0; i < 2; ++i) {
            ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet = set;
            ws[i].dstBinding = static_cast<std::uint32_t>(i);
            ws[i].descriptorCount = 1;
            ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(device_, 2, ws, 0, nullptr);
    };
    auto write_one = [&](VkDescriptorSet set, VkBuffer b0) {
        VkDescriptorBufferInfo dbi{};
        dbi.buffer = b0; dbi.offset = 0; dbi.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet ws{};
        ws.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws.dstSet = set;
        ws.dstBinding = 0;
        ws.descriptorCount = 1;
        ws.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(device_, 1, &ws, 0, nullptr);
    };

    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t log2_n_u = static_cast<std::uint32_t>(stages);

    // bitrev: buf_a -> buf_b
    {
        struct { std::uint32_t n; std::uint32_t log2_n; } push{n_u, log2_n_u};
        write_two(sets[0], plan.buf_a, plan.buf_b);
        dispatch_one(pipe_bitrev_, pl_bitrev_, sets[0], &push, sizeof(push),
                     static_cast<std::uint32_t>((n + 63) / 64));
    }
    // In-place inverse FFT stages on buf_b
    for (std::size_t s = 0; s < stages; ++s) {
        write_one(sets[1 + s], plan.buf_b);
        struct { std::uint32_t n; std::uint32_t m_half; int dir; std::uint32_t pad; } push;
        push.n = n_u;
        push.m_half = static_cast<std::uint32_t>(1u << s);
        push.dir = -1;
        push.pad = 0;
        dispatch_one(pipe_fft_, pl_fft_, sets[1 + s], &push, sizeof(push),
                     static_cast<std::uint32_t>(((n / 2) + 63) / 64));
    }
    // c2r_finish: buf_b -> buf_real
    write_two(sets[1 + stages], plan.buf_b, plan.buf_real);
    dispatch_one(pipe_c2r_, pl_c2r_, sets[1 + stages], (void*)&n_u, sizeof(n_u),
                 static_cast<std::uint32_t>((n + 63) / 64));

    // buf_real -> stage
    VkBufferCopy copyback{};
    copyback.size = sizeof(float) * n;
    vkCmdCopyBuffer(cmd_buf_, plan.buf_real, plan.buf_stage, 1, &copyback);

    VK_CHECK(vkEndCommandBuffer(cmd_buf_));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf_;
    VK_CHECK(vkResetFences(device_, 1, &fence_));
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
    VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));

    std::memcpy(output, plan.stage_mapped, n * sizeof(float));

    vkFreeDescriptorSets(device_, desc_pool_,
                         static_cast<std::uint32_t>(sets.size()), sets.data());
}

// ============================================================
// VulkanBackend wrapper
// ============================================================
VulkanBackend::VulkanBackend(bool enable_validation, bool prefer_discrete_gpu)
    : impl_(std::make_unique<VulkanBackendImpl>(enable_validation, prefer_discrete_gpu)) {}

VulkanBackend::~VulkanBackend() = default;

void VulkanBackend::fft_r2c(const float* in, std::complex<float>* out, std::size_t n) {
    impl_->fft_r2c(in, out, n);
}
void VulkanBackend::fft_c2r(const std::complex<float>* in, float* out, std::size_t n) {
    impl_->fft_c2r(in, out, n);
}
void VulkanBackend::fft_r2c_batch(const float* in, std::complex<float>* out,
                                   std::size_t n, std::size_t batch) {
    // Naive batched: just loop. A real implementation would batch via
    // multiple descriptor sets in one command buffer.
    const std::size_t bins = n / 2 + 1;
    for (std::size_t i = 0; i < batch; ++i) {
        fft_r2c(in + i * n, out + i * bins, n);
    }
}
void VulkanBackend::fft_c2r_batch(const std::complex<float>* in, float* out,
                                   std::size_t n, std::size_t batch) {
    const std::size_t bins = n / 2 + 1;
    for (std::size_t i = 0; i < batch; ++i) {
        fft_c2r(in + i * bins, out + i * n, n);
    }
}

const char* VulkanBackend::name() const {
    static thread_local std::string s;
    s = std::string("Vulkan (") + impl_->device_name() + ")";
    return s.c_str();
}
const char* VulkanBackend::device_name() const { return impl_->device_name().c_str(); }
const char* VulkanBackend::driver_info() const { return impl_->driver_info().c_str(); }
bool VulkanBackend::is_software_renderer() const { return impl_->is_software(); }

bool vulkan_backend_available() noexcept {
    try {
        VulkanBackend b(false, true);
        (void)b;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace aboba
