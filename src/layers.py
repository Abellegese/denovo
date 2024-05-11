"""
Implementation of the Encoder are Adapted from the Instanovo
Github:

"""


import flax.linen as nn
import jax.numpy as jnp
from utils import *
import jax
from flax import struct
from typing import Any
import numpy as np

@struct.dataclass
class Config:
  dtype: Any = jnp.bfloat16

class PositionalEncoding(nn.Module):
    d_model: int        
    max_len: int = 5000   

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = np.array(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x

class MultiScalePeakEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding based on Voronov et. al."""

    h_size: int = 512
    dropout: float = 0.0
    train:bool = True

    def setup(self):
        self.mlp = MLP(self.h_size, self.dropout, self.train)
        self.head = Head(self.h_size, self.dropout, self.train)
        freqs = (
            2 * jnp.pi / jnp.logspace(-2, -3, int(self.h_size / 2), dtype=jnp.float64)
        )
        self.freqs = jnp.array(freqs)

    def __call__(self, mz_values, intensities):
        """Encode peaks."""
        x = self.encode_mass(mz_values)
        x = self.mlp(x)
        x = jnp.concatenate([x, intensities], axis=-1)
        x = self.head(x)
        return x

    def encode_mass(self, x):
        """Encode mz."""
        x = self.freqs[None, None, :] * x
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        return x.astype(jnp.float32)


class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    train: bool

    def setup(self):
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_prob,
            deterministic=not self.train, 
            dtype=Config.dtype
        )
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob, deterministic=not self.train),
            nn.relu,
            nn.Dense(self.dim_feedforward),
        ]
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob, deterministic=not self.train)

    def __call__(self, x, mask=None):
        # Attention part
        if mask is not None:
            # expand mask
            mask = expand_mask(mask)
        x = self.self_attn(x, mask=mask)
        x = x + self.dropout(x)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = (
                l(linear_out))
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    train: bool

    def setup(self):
        self.layers = [
            EncoderBlock(
                self.input_dim,
                self.num_heads,
                self.dim_feedforward,
                self.dropout_prob,
                self.train
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

class MLP(nn.Module):
    h_size: int = 512
    dropout: float = 0.0
    train:bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.h_size, dtype=Config.dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not self.train)(x)
        x = nn.Dense(self.h_size, dtype=Config.dtype)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not self.train)(x)
        return x


class Head(nn.Module):
    h_size: int = 512
    dropout: float = 0.0
    train: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.h_size + 1, dtype=Config.dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not self.train)(x)
        x = nn.Dense(self.h_size, dtype=Config.dtype)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not self.train)(x)
        return x


class Encoder(nn.Module):
    # i2s: dict[int, str]
    residues: dict[str, float]
    dim_model: int = 512
    n_head: int = 16
    dim_feedforward: int = 512
    n_layers: int = 9
    dropout: float = 0.1
    max_length: int = 30
    max_charge: int = 5
    bos_id: int = 1
    eos_id: int = 2
    use_depthcharge: bool = True
    dec_precursor_sos: bool = True
    train: bool = True

    def setup(self):
        self.pos_encoding = PositionalEncoding(self.dim_model, 5000)
        self.latent_spectrum = self.param('latent_spectrum', nn.initializers.normal(), (1, 1, self.dim_model))
        self.encoder = TransformerEncoder(
            num_layers=self.n_layers,
            input_dim=self.dim_model,
            num_heads=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout_prob=self.dropout,
            train=self.train
        )

        if not self.dec_precursor_sos:
            self.peak_encoder = MultiScalePeakEmbedding(
                self.dim_model, dropout=self.dropout, train=self.train
            )
            self.mass_encoder = self.peak_encoder.encode_mass
            self.charge_encoder = nn.Embed(
                self.max_charge, self.dim_model, dtype=Config.dtype
            )

    @nn.compact
    def __call__(self, x, p, x_mask):
        return self._encoder(x, p, x_mask)
    def _encoder(self, x, p, x_mask):
        if x_mask is None:
            x_mask = ~x.sum(dim=2).bool()
        # Peak encoding
        if not self.use_depthcharge:
            x = self.peak_encoder(
                jnp.array(x[:, :, [0]]),
                jnp.array(x[:, :, [1]]),
            )
        else:
            x = self.peak_encoder(x)
        x = self.pos_encoding(x)
        # x = self.peak_norm(x)
        # Self-attention on latent spectra AND peaks
        # fmt: off
        x = jnp.concatenate(
            (
                jnp.repeat(self.latent_spectrum, x.shape[0], axis=0), 
                x
            ),
            axis=1,
        )
        # fmt: on
        latent_mask = jnp.zeros((x_mask.shape[0], 1, 1), dtype=bool)
        x_mask = jnp.concatenate([latent_mask, x_mask], axis=1)

        x = self.encoder(x, x_mask)
        if not self.dec_precursor_sos:
            # Prepare precursors
            masses = self.mass_encoder(jnp.array(p[:, None, [0]]))
            charges = self.charge_encoder(jnp.array(p[:, 1], dtype=int) - 1)
            precursors = masses + charges[:, None, :]
            # Concatenate precursors
            x = jnp.concatenate([precursors, x], axis=1)
            prec_mask = jnp.zeros_like(x_mask[:, :1], dtype=bool)
            x_mask = jnp.concatenate([prec_mask, x_mask], axis=1)
        return x, x_mask
