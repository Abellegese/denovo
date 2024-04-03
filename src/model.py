from loss import InfoNCELoss
import jax
import jax.numpy as jnp
import flax.linen as nn
from utils import *
from flax import struct
from typing import Any


@struct.dataclass
class Config:
  dtype: Any = jnp.bfloat16

def _get_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.triu(jnp.ones((seq_len, seq_len))) == 1
    mask = mask.astype(jnp.float32) 
    mask = mask.at[mask == 0].set(-jnp.inf)
    mask = mask.at[mask == 1].set(0.0)
    return jnp.transpose(mask)

class AutoregressiveModel(nn.Module):
    output_dim: int
    n_heads: int

    def setup(self):
        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.output_dim, dtype=Config.dtype
        )
        self.layer_norm = nn.LayerNorm()
        self.fc = nn.Dense(self.output_dim, dtype=Config.dtype)

    @nn.compact
    def __call__(self, x):
        x_mask = _get_causal_mask(x.shape[1])
        out = self.mha(x, mask=x_mask)  # Self attention
        out = self.layer_norm(x + out)  # Add & Norm
        out = self.fc(out)  # Feed forward
        return out

class CPCModel(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    batch_size: int
    encoders: any
    regressor:bool = True
    def setup(self):
        self.encoder = self.encoders
        self.autoregressor = AutoregressiveModel(self.hidden_dim, self.output_dim)
        self.loss = InfoNCELoss(
            self.hidden_dim,
            self.input_dim,
            batch_size=self.batch_size,
            pred_timestep=12,
        )
    def get_latent_representations(self, spectra, precurs, spectr_mask):
        embedding = self.encoder(spectra, precurs, spectr_mask)
        embedding = embedding[0]
        # making it suitable for inference time
        # we dont use the decoder afeter
        if self.regressor:
            context = self.autoregressor(embedding)
            return embedding, context
        return embedding, None

    def __call__(self, spectra, precurs, spectr_mask):
        embedding, context = self.get_latent_representations(spectra, precurs, spectr_mask)
        if self.regressor:
            loss = self.loss(spectra, embedding, context)
            return loss, embedding, context
        return None, embedding, None
