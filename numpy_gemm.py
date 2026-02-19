import numpy as np
import time

m = n = k = 4096

A = np.random.randn(m, k).astype(np.float32)
B = np.random.randn(k, n).astype(np.float32)

# warmup
C = A @ B

start = time.perf_counter()
C = A @ B
end = time.perf_counter()

_time = (end - start)
tflops = (2 * m * n * k) / _time / 1e12

print(f"TFLOPs: {tflops}")
