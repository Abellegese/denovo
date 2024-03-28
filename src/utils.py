import jax.numpy as jnp
import math
import flax.linen as nn
import jax

def expand_mask(mask):
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    # Handle cases based on initial dimensions:
    if mask.ndim == 3:
        # Add a dimension for number of heads (unsqueeze at index 1)
        mask = jnp.expand_dims(mask, axis=1)
    elif mask.ndim == 2:
        # Add dimensions for batch size and number of heads (unsqueeze twice)
        mask = jnp.expand_dims(jnp.expand_dims(mask, axis=0), axis=1)
    # Ensure final shape is (batch_size, num_heads, seq_length, seq_length)
    return mask

def _get_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.triu(jnp.ones((seq_len, seq_len))) == 1
    mask = mask.astype(jnp.float32) 
    mask = mask.at[mask == 0].set(-jnp.inf)
    mask = mask.at[mask == 1].set(0.0)
    return jnp.transpose(mask)


def _count_params(params): return sum(p.size for p in jax.tree_leaves(params))