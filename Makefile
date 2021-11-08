

NVCC = nvcc --compiler-options="-Wall -Wextra -O3" -std=c++14 -arch=compute_52 -code=sm_52


default: cuda_bandwidth_measurement


cuda_bandwidth_measurement: Makefile kernel.cu
	$(NVCC) -o cuda_bandwidth_measurement kernel.cu

clean:
	rm -f cuda_bandwidth_measurement
