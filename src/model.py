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

def make_causal_mask(sql):
  idxs = jnp.arange(sql)
  mask = jnp.where(idxs[:, None] <= idxs[None, :], 0, -1e9)
  return mask

class AutoregressiveModel(nn.Module):
    output_dim: int
    n_heads: int
    train: bool = True
    dropout: float = 0.1

    def setup(self):
        self.lm1 = nn.LayerNorm()
        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            deterministic=not self.train,
            dropout_rate=self.dropout,
            qkv_features=self.output_dim,
            dtype=Config.dtype
        )
        self.fc = nn.Dense(self.output_dim, dtype=Config.dtype)

    @nn.compact
    def __call__(self, x):
        x_mask = make_causal_mask(x.shape[1])
        out = self.mha(x, mask=x_mask)  # Self attention
        out = self.lm1(x + out)  # Add & Norm
        out = self.fc(out)  # Feed forward
        return out
class CPCModel(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    batch_size: int
    encoders: any
    regressor:bool = True
    train:bool = True
    dropout:float = 0.1
    def setup(self):
        self.encoder = self.encoders
        self.autoregressor = AutoregressiveModel(self.hidden_dim, self.output_dim, self.train, self.dropout)
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
