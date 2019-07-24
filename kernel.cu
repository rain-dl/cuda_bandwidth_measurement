
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

auto measure_host_to_device_memcopy_mb(const int n, const bool enable_async)
{
	constexpr long long mb=1<<20;
	const int size=n*mb/(2*sizeof(int));

	int *host;
	cudaMallocHost((void **)&host, size);

	int *device;
	cudaMalloc((void **)&device, size);

	std::chrono::time_point<std::chrono::system_clock> start, end;

	if(!enable_async)
	{
		start=std::chrono::system_clock::now();
		cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
		cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
		end=std::chrono::system_clock::now();
	}
	else
	{
		start=std::chrono::system_clock::now();
		cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost);
		end=std::chrono::system_clock::now();
	}

	cudaFreeHost(host);
	cudaFree(device);

	return end-start;
}


int main()
{
	for(const auto n :{16, 32, 64, 128, 256, 512, 1024, 2048})
	{
		int sum=0;

		for(int i=0; i<5; ++i)
		{
			const auto time=std::chrono::duration_cast<std::chrono::microseconds>(measure_host_to_device_memcopy_mb(n, false)).count();
			std::cout<<"@sync_copy size: "<<n<<"MB, time="<<time<<std::endl;
			sum+=time;
		}

		std::cout<<"avg="<<sum/5<<std::endl;
	}

	return 0;
}

