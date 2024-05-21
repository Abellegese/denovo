from loss import InfoNCELoss
import jax
import jax.numpy as jnp
import flax.linen as nn
from utils import *
from flax import struct
from typing import Any
from layers import PositionalEncoding
@struct.dataclass
class Config:
    dtype: Any = jnp.bfloat16

def make_causal_mask(sql):
    idxs = jnp.arange(sql)
    mask = jnp.where(idxs[:, None] <= idxs[None, :], 0, -1e9)
    return mask

class AutoregressiveModel(nn.Module):
    output_dim: int
    n_heads: int = 8
    dim_feedforward: int = 128
    train: bool = True
    num_layers: int = 4
    dropout:float = 0.1

    def setup(self):
        # Define the decoder block
        self.decoder_blocks = [
            DecoderBlock(
                output_dim=self.output_dim,
                n_heads=self.n_heads,
                dim_feedforward=self.dim_feedforward,
                train=self.train,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    output_dim: int
    n_heads: int
    dim_feedforward: int
    train: bool = True
    dropout: float = 0.1
    def setup(self):
        self.pos_encoding = PositionalEncoding(self.output_dim, 5000)
        # Two-layer MLP
        self.linf = nn.Dense(self.output_dim)
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(
                rate=self.dropout,
                deterministic=not self.train
                ),
            nn.relu,
            nn.Dense(self.dim_feedforward),
        ]
        self.lm1 = nn.LayerNorm()
        self.lm2 = nn.LayerNorm()
        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            deterministic=not self.train,
            dropout_rate=self.dropout
        )

    def __call__(self, x):
        x_mask = make_causal_mask(x.shape[1])
        x = self.pos_encoding(x)
        out = self.mha(x, mask=x_mask)
        out = self.lm1(x + out)
        linear_out = out
        for layer in self.linear:
            linear_out = (
                layer(linear_out)
                if not isinstance(layer, nn.Dropout)
                else layer(linear_out)
            )
        out = out + linear_out
        out = self.lm2(out)
        out = self.linf(out)
        return out
# class GRUBlock(nn.Module):
#     hidden_dim: int
#     dropout: float = 0.1
#     train: bool = True

#     def setup(self):
#         self.gru = nn.GRUCell(features=self.hidden_dim)
#         self.dropout_layer = nn.Dropout(rate=self.dropout, deterministic=not self.train)
#         self.lm1 = nn.LayerNorm()

#     def __call__(self, x, carry):
#         carry, x = self.gru(carry, x)
#         # x = self.lm1(x)
#         # x = self.dropout_layer(x)
#         return carry, x

# class AutoregressiveModel(nn.Module):
#     hidden_dim: int
#     num_layers: int = 4
#     dropout: float = 0.5
#     train: bool = True

#     def setup(self):
#         # Define the GRU blocks
#         self.gru_blocks = [
#             GRUBlock(
#                 hidden_dim=self.hidden_dim,
#                 dropout=self.dropout,
#                 train=self.train,
#             )
#             for _ in range(self.num_layers)
#         ]

#     def __call__(self, x):
#         # Initialize carry (hidden state) with zeros
#         carry = jnp.zeros((x.shape[0], x.shape[1], self.hidden_dim))
#         for block in self.gru_blocks:
#             carry, x = block(x, carry)
#         return x

class CPCModel(nn.Module):
    input_dim: int
    hidden_dim: int
    num_head:int
    output_dim: int
    dim_feedforward:int
    batch_size: int
    encoders: any
    num_layers: int = 2
    regressor: bool = True
    train: bool = True
    dropout: float = 0.1

    def setup(self):
        self.encoder = self.encoders
        self.autoregressor = AutoregressiveModel(
            output_dim=self.hidden_dim, 
            train=self.train, 
            num_layers=self.num_layers, 
            dropout=self.dropout
        )
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
        embedding, context = self.get_latent_representations(
            spectra, precurs, spectr_mask
        )
        if self.regressor:
            loss, logits = self.loss(spectra, embedding, context)
            return loss, logits, embedding, context
        return None, embedding, None
