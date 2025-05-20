#pragma once
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <nvml.h>
#include <sstream>
#include <thread>
#include <vector>

namespace getWatt {

double get_current_power(const unsigned gpu_id) {
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(gpu_id, &device);
    unsigned int power;
    nvmlDeviceGetPowerUsage(device, &power);
    return power / 1000.0;
}

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

    int gpu_id = 0;

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

        const auto power = get_current_power(gpu_id);

        const auto end_clock_1    = std::chrono::high_resolution_clock::now();
        const auto elapsed_time_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_clock_1 - start_clock).count();

        profiling_result.push_back(PowerProfile{power, elapsed_time});

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(std::chrono::milliseconds(std::max<std::time_t>(static_cast<int>(interval) * count, elapsed_time_1) - elapsed_time_1));
        count++;
    } while (semaphore);

    thread.join();

    nvmlShutdown();

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
