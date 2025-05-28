#pragma once
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#if defined(__NVCC__)
#include <nvml.h>
#elif defined(__HIPCC__)
#include <amd_smi/amdsmi.h>
#include <functional>
#endif
#include <sstream>
#include <thread>
#include <vector>

namespace getWatt {

#if defined(__NVCC__)
double get_current_power(const unsigned gpu_id) {
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(gpu_id, &device);
    unsigned int power;
    nvmlDeviceGetPowerUsage(device, &power);
    return power / 1000.0;
}
#elif defined(__HIPCC__)
#define CHK_AMDSMI_RET(RET)                                                    \
    {                                                                          \
        if (RET != AMDSMI_STATUS_SUCCESS) {                                    \
            const char *err_str;                                               \
            std::cout << "AMDSMI call returned " << RET << " at line "         \
                      << __LINE__ << std::endl;                                \
            amdsmi_status_code_to_string(RET, &err_str);                       \
            std::cout << err_str << std::endl;                                 \
            return profiling_result;                                           \
        }                                                                      \
    }

double get_current_power(amdsmi_processor_handle handle) {
    amdsmi_power_info_t info;
    amdsmi_get_power_info(handle, &info);
    return static_cast<double>(info.average_socket_power);
}
#endif

struct PowerProfile {
    double power;
    std::time_t timestamp;
};

double get_elapsed_time(const std::vector<PowerProfile> &profiling_data_list) {
    if (profiling_data_list.size() == 0) {
        return 0.0;
    }
    return (profiling_data_list[profiling_data_list.size() - 1].timestamp - profiling_data_list[0].timestamp) * 1.e-6;
}

std::vector<PowerProfile> getGpuPowerUsage(const std::function<void(void)> func, const std::time_t interval) {
    std::vector<PowerProfile> profiling_result;

    int gpu_id = 0;

#if defined(__NVCC__)
    nvmlReturn_t result;

    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
    }

    unsigned int deviceCount;
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to device count: " << nvmlErrorString(result) << std::endl;
    }
#elif defined(__HIPCC__)
    amdsmi_status_t ret;
    ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    CHK_AMDSMI_RET(ret)

    // Get the socket count available for the system.
    uint32_t socket_count = 0;
    ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    CHK_AMDSMI_RET(ret)

    // Allocate the memory for the sockets
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    // Get the sockets of the system
    ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
    CHK_AMDSMI_RET(ret)

    int i=0;
    // Get the device count available for the socket.
    uint32_t device_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
    CHK_AMDSMI_RET(ret)

    // Allocate the memory for the device handlers on the socket
    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    // Get all devices of the socket
    ret = amdsmi_get_processor_handles(sockets[i],
                                    &device_count, &processor_handles[0]);
    CHK_AMDSMI_RET(ret)

    amdsmi_processor_handle gpu_handle = processor_handles[gpu_id];
#endif

    unsigned count = 0;

    int semaphore = 1;

    std::thread thread([&]() {
        func();
        semaphore = 0;
    });

    const auto start_clock = std::chrono::high_resolution_clock::now();
    do {
        const auto end_clock    = std::chrono::high_resolution_clock::now();
        const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count();

#if defined(__NVCC__)
        const auto power = get_current_power(gpu_id);
#elif defined(__HIPCC__)
        const auto power = get_current_power(gpu_handle);
#endif

        const auto end_clock_1    = std::chrono::high_resolution_clock::now();
        const auto elapsed_time_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_clock_1 - start_clock).count();

        profiling_result.push_back(PowerProfile{power, elapsed_time});

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(std::chrono::milliseconds(std::max<std::time_t>(static_cast<int>(interval) * count, elapsed_time_1) - elapsed_time_1));
        count++;
    } while (semaphore);

    thread.join();

#if defined(__NVCC__)
    nvmlShutdown();
#elif defined(__HIPCC__)
    amdsmi_shut_down();
#endif

    return profiling_result;
}

double get_integrated_power_consumption(const std::vector<PowerProfile> &profiling_data_list) {
    if (profiling_data_list.size() == 0) {
        return 0.0;
    }

    double power_consumption = 0.;
    for (unsigned i = 1; i < profiling_data_list.size(); i++) {
        const auto elapsed_time = (profiling_data_list[i].timestamp - profiling_data_list[i - 1].timestamp) * 1e-6;
        // trapezoidal integration
        power_consumption += (profiling_data_list[i].power + profiling_data_list[i - 1].power) / 2 * elapsed_time;
    }
    return power_consumption;
}

//=================================================================
// Function returns power consumption Watt
//================================================================
std::vector<double> getWatt(const std::function<void(void)> func, const size_t m, const size_t n, const size_t k) {
    constexpr size_t duration_time = 10;
    size_t cnt                     = 0;
    std::vector<PowerProfile> powerUsages;
    powerUsages = getGpuPowerUsage(
        [&]() {
            gpuDeviceSynchronize();
            const auto start_clock = std::chrono::system_clock::now();
            while (true) {
                func();
                if (((++cnt) % 10) == 0) {
                    gpuDeviceSynchronize();
                    const auto current_clock = std::chrono::system_clock::now();
                    const auto elapsed_time =
                        std::chrono::duration_cast<std::chrono::microseconds>(current_clock - start_clock).count() * 1e-6;
                    if (elapsed_time > duration_time) {
                        break;
                    }
                }
            }
        },
        100);
    const double power          = get_integrated_power_consumption(powerUsages);
    const double elapsed_time   = get_elapsed_time(powerUsages);
    const double watt           = power / elapsed_time;
    const double flops_per_watt = 2.0 * m * n * k * cnt / power;
    std::vector<double> results{watt, flops_per_watt};
    return results;
}

} // namespace getWatt
