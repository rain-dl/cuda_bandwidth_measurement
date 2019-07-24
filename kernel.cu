
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

auto measure_host_to_device_memcopy_mb(const int n, const bool enable_async)
{
	constexpr long long mb=1<<20;
	const int size=n*mb/(2*sizeof(int));

	int *host;
	cudaMallocHost((void **)&host, size);

	int *device;
	cudaMalloc((void **)&device, size);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	if(!enable_async)
	{
		cudaEventRecord(start);
		cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
		cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop);
	}
	else
	{
		cudaEventRecord(start);
		cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop);
	}

	cudaEventSynchronize(stop);

	cudaFreeHost(host);
	cudaFree(device);

	float time=0;
	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
}


int main()
{
	for(const auto n :{16, 32, 64, 128, 256, 512, 1024, 2048})
	{
		float sum=0;

		for(int i=0; i<5; ++i)
		{
			const auto time=measure_host_to_device_memcopy_mb(n, false);
			std::cout<<"@sync_copy size: "<<n<<"MB, time="<<time<<std::endl;
			sum+=time;
		}

		std::cout<<"avg="<<sum/5<<"s"<<std::endl;
	}

	return 0;
}

