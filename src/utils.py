import jax.numpy as jnp
import math
import flax.linen as nn


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    # check for improvement
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


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
