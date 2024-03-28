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
x = jnp.ones((1, 4, 4))
decoder_mask = nn.combine_masks(
        nn.make_attention_mask(
            jnp.ones_like(x),
            x != 1, 
            dtype=bool),
        nn.make_causal_mask(x, 
                            dtype=bool)
    ) 
print(decoder_mask)