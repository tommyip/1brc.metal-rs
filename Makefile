gpu_bandwidth_limit: src/bin/gpu_bandwidth_limit/kernel.metal
	xcrun -sdk macosx metal -c src/bin/gpu_bandwidth_limit/kernel.metal -o src/bin/gpu_bandwidth_limit/kernel.metalar
	xcrun -sdk macosx metallib src/bin/gpu_bandwidth_limit/kernel.metalar -o src/bin/gpu_bandwidth_limit/kernel.metallib
