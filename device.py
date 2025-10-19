import jax
import jax.numpy as jnp

# Set the default device to Metal
jax.config.update('jax_platform_name', 'METAL')

# All subsequent JAX operations will run on the Metal GPU
x = jnp.arange(10)
print(x.device())
