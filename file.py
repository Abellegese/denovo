import jax
import jax.numpy as jnp
import torch
import jax.numpy as jnp
import flax.linen as nn
def _get_causal_mask(seq_len: int) -> jnp.ndarray:
    """Creates a causal mask in JAX.

    Args:
        seq_len: The length of the sequence.

    Returns:
        A jnp.ndarray of shape (seq_len, seq_len) with values of 0 for masked
        elements and -inf for unmasked elements.
    """

    mask = jnp.triu(jnp.ones((seq_len, seq_len))) == 1
    mask = mask.astype(jnp.float32)  # Ensure float32 for -inf compatibility
    mask = mask.at[mask == 0].set(-jnp.inf)
    mask = mask.at[mask == 1].set(0.0)  # Correctly set masked elements to 0
    return jnp.transpose(mask)  # Match PyTorch output order

    
import jax.numpy as jnp
import numpy as np
import torch

# Create dummy Torch tensors
spectra = torch.randn((10, 5))
precursors = torch.randn((10, 3))
spectra_mask = torch.randn((10, 5))

# Define the conversion function
def torch_to_jax(tensor):
    return jnp.array(tensor.detach().cpu().numpy()).astype(jnp.bfloat16)

# Convert Torch tensors to JAX arrays
spectra_jax = torch_to_jax(spectra)
precursors_jax = torch_to_jax(precursors)
spectra_mask_jax = torch_to_jax(spectra_mask)

# Print the converted arrays
print("Spectra (JAX):", spectra_jax)
print("Precursors (JAX):", precursors_jax)
print("Spectra Mask (JAX):", spectra_mask_jax)
