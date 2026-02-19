import jax
import jax.numpy as jnp
import time

key = jax.random.PRNGKey(0)

m = n = k = 4096
A = jax.random.normal(key, (m, k))
B = jax.random.normal(key, (k, n))

# warmup
C = A @ B
C.block_until_ready()

start = time.perf_counter()
C = A @ B
C.block_until_ready() 
end = time.perf_counter()

_time = (end - start)
tflops = (2 * m * n * k) / _time / 1e12

print(f"TFLOPs: {tflops}")
