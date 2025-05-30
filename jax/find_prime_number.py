"""

Compare prime-checking on CPU using NumPy vs JAX.

"""
import time
import sys
import numpy as np
import jax
import jax.numpy as jnp

def is_prime(n):
    if n < 2:
        return False 
    divs = np.arange(2, int(np.sqrt(n)) + 1)
    # print(not np.any(n%divs == 0))
    return not np.any(n % divs == 0)

def prime_numpy(n):
    
    if n < 2:
        return n 
    
    start = n
    # while loop until finding prime n
    while not is_prime(start):
        start += 1
        print(f"Currently at: {start}")        

    smallest_prime_after_n = start
    return smallest_prime_after_n 

@jax.jit
def prime_jax(n: jnp.ndarray):
    
    
    """Check primality by divisibility up to sqrt(n) using JAX on CPU."""
    # special-case small n
    def _check(x):
        if x < 2:
            return False
        # build array 2..sqrt(x)
        limit = jnp.floor(jnp.sqrt(x)).astype(jnp.int32)
        divs = jnp.arange(2, limit + 1)
        # check any zero remainder
        return jnp.all(jnp.mod(x, divs) != 0)
    return _check(n)

def main():
    user_input = input("Find a prime number bigger than: ")
    print(f"You entered: {user_input}")

    try:
        n = int(user_input)
    except ValueError:
        print("Please provide a valid integer.")
        sys.exit(1)

    print(f"Testing n = {n}\n")

    # NumPy version
    start_time = time.time()
    result_np = prime_numpy(n)
    end_time = time.time()
    print(f"NumPy says:   {result_np}")
    print(f"Numpy time: {end_time - start_time}")
    # # JAX version (compile & run)
    # # wrap n in a 0-d array for JAX
    # n_jax = jnp.array(n, dtype=jnp.int32)
    # result_jax = bool(prime_jax(n_jax))
    # print(f"JAX   says:     {n} is {'prime' if result_jax else 'not prime'}")

if __name__ == "__main__":
    main()