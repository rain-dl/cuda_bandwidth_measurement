#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#ifdef __GNUC__
#include <getopt.h>
#else
#include "getopt.win.h"
#endif
#ifdef _WIN32
#include <Windows.h>
#endif

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

auto measure_host_device_bandwidth_mb(const int block_size, const int loop_count, const bool enable_sync)
{
	int *host, *device;
	cudaMallocHost((void **)&host, block_size);
	cudaMalloc((void **)&device, block_size);

	cudaEvent_t hd_start, hd_stop, dh_start, dh_stop;
	cudaEventCreate(&hd_start);
	cudaEventCreate(&hd_stop);
	cudaEventCreate(&dh_start);
	cudaEventCreate(&dh_stop);

	float hd_sum = 0, dh_sum = 0;

	for (int i = 0; i < loop_count; i++)
	{
		float hd_time, dh_time;

		if (enable_sync)
		{
			cudaEventRecord(hd_start);
			cudaMemcpy(device, host, block_size, cudaMemcpyHostToDevice);
			cudaEventRecord(hd_stop);
			cudaEventSynchronize(hd_stop);

			cudaEventRecord(dh_start);
			cudaMemcpy(host, device, block_size, cudaMemcpyDeviceToHost);
			cudaEventRecord(dh_stop);
			cudaEventSynchronize(dh_stop);
		}
		else
		{
			cudaEventRecord(hd_start);
			cudaMemcpyAsync(device, host, block_size, cudaMemcpyHostToDevice);
			cudaEventRecord(hd_stop);
			cudaEventSynchronize(hd_stop);

			cudaEventRecord(dh_start);
			cudaMemcpyAsync(host, device, block_size, cudaMemcpyDeviceToHost);
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

void measure_host_device_bandwidth(const int block_size, const int multiple_start, const int multiple_end, const int loop_count, const bool enable_sync)
{
	std::cout << "host <-> device  " << (enable_sync ? "sync" : "aysnc") << std::endl;
	std::cout << "data size[MB], host to device[ms], host to device[GB/s], device to host[ms], device to host[GB/s]" << std::endl;
	for (int n = multiple_start; n <= multiple_end; n++)
	{
		const auto time = measure_host_device_bandwidth_mb(block_size * n, loop_count, enable_sync);
		float hd_sum = time.first;
		float dh_sum = time.second;
		std::cout << block_size * n / 1024 / 1024 << ", " << hd_sum / loop_count << ", " << (float)block_size * n * loop_count * 1000 / hd_sum / 1024 / 1024 / 1024 << ", "
			<< dh_sum / loop_count << ", " << (float)block_size * n * loop_count * 1000 / dh_sum / 1024 / 1024 / 1024 << std::endl;
	}
	std::cout << "--\n" << std::endl;
}

auto measure_global_bandwidth_kb(const int block_size, const int loop_count)
{
	int *device1, *device2;
	cudaMalloc((void **)&device1, block_size);
	cudaMalloc((void **)&device2, block_size);

	float sum = 0, time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 grid(1);
	dim3 threads(1);

	for (int i = 0; i < loop_count; i++)
	{
		cudaEventRecord(start);
		cudaMemcpyAsync(device1, device2, block_size, cudaMemcpyDeviceToDevice);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&time, start, stop);
		sum += time;
	}

	cudaFree(device1);
	cudaFree(device2);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
}

void measure_global_bandwidth(const int block_size, const int multiple_start, const int multiple_end, const int loop_count)
{
	std::cout << "global memory" << std::endl;
	std::cout << "data size[MB], time[ms], bandwidth[GB/s]" << std::endl;

	for (int n = multiple_start; n <= multiple_end; n++)
	{
		auto sum = measure_global_bandwidth_kb(block_size * n, loop_count);
		std::cout << block_size * n / 1024 / 1024 << ", " << sum / loop_count << ", " << (float)block_size * n * loop_count * 1000 / sum / 1024 / 1024 / 1024 << std::endl;
	}
	std::cout << "--\n" << std::endl;
}

void usage(const char* program_name)
{
	std::cout << "usage: " << program_name << " [options]" << std::endl << std::endl
		<< "Options:" << std::endl
		<< "    -b --block_size <size>       memory block size, default is 10MB" << std::endl
		<< "    -m --start <multiple start>  loop count, default is 1" << std::endl
		<< "    -n --end <multiple end>      loop count, default is 10" << std::endl
		<< "    -l --loop <loop count>       loop count, default is 100" << std::endl
		<< "    -s --sync                    test with sync mode" << std::endl
		<< "    --help                       print this help" << std::endl
		<< std::endl;
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
	setlocale(LC_ALL, "chs");
#endif

	size_t block_size = static_cast<size_t>(10 * 1024) * 1024;
	int multiple_start = 1, multiple_end = 10;
	int loop_count = 100;
	bool enable_sync = false;

	while (1)
	{
		static struct option long_options[] = {
			{ "block_size",       required_argument, 0, 'b' },
			{ "multiple_start",   required_argument, 0, 'm' },
			{ "multiple_end",     required_argument, 0, 'n' },
			{ "loop",             required_argument, 0, 'l' },
			{ "sync",             required_argument, 0, 's' },
			{ "help",             no_argument,       0,  0  },
			{ 0,                  0,         0,  0  }
		};
		int option_index;
		auto c = getopt_long(argc, (char* const*)argv, "b:m:n:l:s", long_options, &option_index);
		if (c < 0)
			break;

		switch (c)
		{
		case 'b':
			block_size = _atoi64(optarg);
			break;
		case 'm':
			multiple_start = atoi(optarg);
			break;
		case 'n':
			multiple_end = atoi(optarg);
			break;
		case 'l':
			loop_count = atoi(optarg);
			break;
		case 's':
			enable_sync = true;
			break;
		default:
			usage(argv[0]);
			return 0;
		}
	}

	measure_host_device_bandwidth(block_size, multiple_start, multiple_end, loop_count, enable_sync);
	measure_global_bandwidth(block_size, multiple_start, multiple_end, loop_count);
	return 0;
}
