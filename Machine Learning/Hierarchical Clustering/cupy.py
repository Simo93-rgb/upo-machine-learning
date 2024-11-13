import time
import numpy as np
import cupy as cp


print(cp.cuda.runtime.getDeviceCount())
print(cp.cuda.runtime.getDeviceProperties(0))

# NumPy
a_np = np.random.rand(10000, 10000)
b_np = np.random.rand(10000, 10000)
start = time.time()
c_np = np.dot(a_np, b_np)
end = time.time()
print(f"NumPy time: {end - start}")

# CuPy
a_cp = cp.random.rand(10000, 10000)
b_cp = cp.random.rand(10000, 10000)
start = time.time()
c_cp = cp.dot(a_cp, b_cp)
cp.cuda.Stream.null.synchronize()
end = time.time()
print(f"CuPy time: {end - start}")