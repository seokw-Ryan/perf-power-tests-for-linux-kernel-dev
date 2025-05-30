# This script demonstrates a simple matrix multiplication using JAX.
# It defines a function to perform the multiplication.

import numpy as np
import jax
import jax.numpy as jnp

def simple_mat_mul(a, b):
    """
    Perform matrix multiplication of two matrices a and b using JAX.

    Args:
        a (jax.numpy.ndarray): First matrix.
        b (jax.numpy.ndarray): Second matrix.

    Returns:
        jax.numpy.ndarray: Result of the matrix multiplication.
    """
    return jnp.dot(a, b)


def estimate_jax_array_size(shape, dtype=jnp.float32):
    itemsize = np.dtype(dtype).itemsize  # NumPy handles dtype sizes reliably
    size_in_bytes = np.prod(shape) * itemsize
    size_in_mb = size_in_bytes / (1024 ** 2)
    size_in_gb = size_in_bytes / (1024 ** 3)
    return size_in_bytes, size_in_mb, size_in_gb


def main():
    # size of the matrices
    x = 10000000 

    # Estimate the size of the arrays
    size_bytes, size_mb, size_gb = estimate_jax_array_size((x, x))
    print(f"Estimated size of each array: {size_bytes} bytes, {size_mb:.2f} MB, {size_gb:.2f} GB")

    # Test the limit of the JAX array size before failing
    try:
        x = 1000000000  # Initial size of the matrix
        while True:
            # Size of the array
            x += 10000000
            # Example matrices
            key1 = jax.random.PRNGKey(0)
            key2 = jax.random.PRNGKey(1)
            n = jax.numpy.array(jax.random.uniform(key1, shape=(x, x)))
            m = jax.numpy.array(jax.random.uniform(key2, shape=(x, x)))



            # print("Matrix n: ")
            # print(n)
            # print("Matrix m: ")
            # print(m)

            # Perform matrix multiplication
            result = simple_mat_mul(n, m)
            
            # print("Result of matrix multiplication:")
            print(f"size of the array: {x} and result: {result.shape}")
    except:
        print(f"Failed to allocate array of size {x} x {x} due to memory constraints.")

if __name__ == "__main__":
    main()