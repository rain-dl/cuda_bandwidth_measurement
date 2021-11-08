﻿#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

namespace kernel
{

	__global__ void measure_global_bandwidth_kb(int *out, int *device, int size)
	{
		int r = 0;
		for (int i = 0; i < size; ++i)
		{
			r += device[i];
		}
		*out = r;
	}

}

auto measure_host_device_bandwidth_mb(const int n, const int repeat, const bool enable_sync)
{
	const int bytes = n * (1 << 20);

	int *host, *device;
	cudaMallocHost((void **)&host, bytes);
	cudaMalloc((void **)&device, bytes);

	cudaEvent_t hd_start, hd_stop, dh_start, dh_stop;
	cudaEventCreate(&hd_start);
	cudaEventCreate(&hd_stop);
	cudaEventCreate(&dh_start);
	cudaEventCreate(&dh_stop);

	float hd_sum = 0, dh_sum = 0;

	for (int i = 0; i < repeat; i++)
	{
		float hd_time, dh_time;

		if (enable_sync)
		{
			cudaEventRecord(hd_start);
			cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
			cudaEventRecord(hd_stop);
			cudaEventSynchronize(hd_stop);

			cudaEventRecord(dh_start);
			cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost);
			cudaEventRecord(dh_stop);
			cudaEventSynchronize(dh_stop);
		}
		else
		{
			cudaEventRecord(hd_start);
			cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice);
			cudaEventRecord(hd_stop);
			cudaEventSynchronize(hd_stop);

			cudaEventRecord(dh_start);
			cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost);
			cudaEventRecord(dh_stop);
			cudaEventSynchronize(dh_stop);
		}

		cudaEventElapsedTime(&hd_time, hd_start, hd_stop);
		cudaEventElapsedTime(&dh_time, dh_start, dh_stop);

		hd_sum += hd_time;
		dh_sum += dh_time;
	}

	cudaFreeHost(host);
	cudaFree(device);

	cudaEventDestroy(hd_start);
	cudaEventDestroy(hd_stop);
	cudaEventDestroy(dh_start);
	cudaEventDestroy(dh_stop);

	return std::make_pair(hd_sum, dh_sum);
}

void measure_host_device_bandwidth(const bool enable_sync)
{
	const int repeat = 100;
	std::cout << "host <-> device  " << (enable_sync ? "sync" : "aysnc") << std::endl;
	std::cout << "data size[MB], host to device[ms], host to device[GB/s], device to host[ms], device to host[GB/s]" << std::endl;
	for (int n = 32; n <= 256; n += 32)
	{
		const auto time = measure_host_device_bandwidth_mb(n, repeat, enable_sync);
		float hd_sum = time.first;
		float dh_sum = time.second;
		std::cout << n << ", " << hd_sum / repeat << ", " << n * 1000 * repeat / hd_sum / 1024 << ", " << dh_sum / repeat << ", " << n * 1000 * repeat / dh_sum / 1024 << std::endl;
	}
	std::cout << "--\n"
			  << std::endl;
}

auto measure_global_bandwidth_kb(const int n)
{
	const int bytes = n * (1 << 20);

	int *out, *device;
	cudaMalloc((void **)&out, sizeof(int));
	cudaMalloc((void **)&device, bytes);

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 grid(1);
	dim3 threads(1);

	cudaEventRecord(start);
	kernel::measure_global_bandwidth_kb<<<grid, threads>>>(out, device, bytes / sizeof(int));
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	cudaFree(out);
	cudaFree(device);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
}

void measure_global_bandwidth()
{
	const int repeat = 100;
	std::cout << "global memory" << std::endl;
	std::cout << "data size[MB], time[ms], bandwidth[GB/s]" << std::endl;
	for (int n = 32; n <= 512; n += 32)
	{
		float sum = 0;
		for (int i = 0; i < repeat; ++i)
		{
			sum += measure_global_bandwidth_kb(n);
		}

		std::cout << n << ", " << sum / repeat << ", " << n * 1000 * repeat / sum / 1024 << std::endl;
	}
	std::cout << "--\n"
			  << std::endl;
}

int main()
{
	measure_host_device_bandwidth(true);
	measure_host_device_bandwidth(false);
	measure_global_bandwidth();
	return 0;
}
