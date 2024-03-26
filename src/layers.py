import flax.linen as nn
import jax.numpy as jnp
from utils import *


class MultiScalePeakEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding based on Voronov et. al."""

    h_size: int = 10
    dropout: float = 0.0

    def setup(self):
        self.mlp = MLP(self.h_size, self.dropout)
        self.head = Head(self.h_size, self.dropout)
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

    def setup(self):
        # Attention layer
        # self.self_attn = MultiheadAttention(
        #     embed_dim=self.input_dim, num_heads=self.num_heads
        # )
        self.self_attn = nn.MultiHeadDotProductAttention(
            qkv_features=self.input_dim, num_heads=self.num_heads
        )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim),
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        if mask is not None:
            # expand mask
            mask = expand_mask(mask)
        x = self.self_attn(x, mask=mask)
        x = x + self.dropout(x, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.layers = [
            EncoderBlock(
                self.input_dim,
                self.num_heads,
                self.dim_feedforward,
                self.dropout_prob,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)
        return attention_maps


class MLP(nn.Module):
    h_size: int = 10
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.h_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.1, deterministic=False)(x)
        x = nn.Dense(self.h_size)(x)
        x = nn.Dropout(rate=0.1, deterministic=False)(x)
        return x


class Head(nn.Module):
    h_size: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.h_size + 1)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=False)(x)
        x = nn.Dense(self.h_size)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=False)(x)
        return x


class Encoder(nn.Module):
    # i2s: dict[int, str]
    residues: dict[str, float]
    dim_model: int = 768
    n_head: int = 16
    dim_feedforward: int = 2048
    n_layers: int = 9
    dropout: float = 0.1
    max_length: int = 30
    max_charge: int = 5
    bos_id: int = 1
    eos_id: int = 2
    use_depthcharge: bool = True
    dec_precursor_sos: bool = True

    def setup(self):
        self.latent_spectrum = self.param(
            "latent_spectrum",
            nn.initializers.zeros,
            (1, 1, self.dim_model),
        )

        self.encoder = TransformerEncoder(
            num_layers=self.n_layers,
            input_dim=self.dim_model,
            num_heads=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout_prob=self.dropout,
        )

        if not self.dec_precursor_sos:
            self.peak_encoder = MultiScalePeakEmbedding(
                self.dim_model, dropout=self.dropout
            )
            self.mass_encoder = self.peak_encoder.encode_mass
            self.charge_encoder = nn.Embed(self.max_charge, self.dim_model)

    @nn.compact
    def __call__(self, x, p, x_mask):
        return self._encoder(x, p, x_mask)

    def _encoder(self, x, p, x_mask):
        if x_mask is None:
            x_mask = ~x.sum(dim=2).bool()
        # Peak encoding
        if not self.use_depthcharge:
            x = self.peak_encoder(
                jnp.array(x[:, :, [0]].numpy()),
                jnp.array(x[:, :, [1]].numpy()),
            )
        else:
            x = self.peak_encoder(x)
        # x = self.peak_norm(x)
        # Self-attention on latent spectra AND peaks
        latent_spectra = jnp.broadcast_to(
            self.latent_spectrum,
            (x.shape[0],) + self.latent_spectrum.shape[1:],
        )
        x = jnp.concatenate([latent_spectra, x], axis=1)
        latent_mask = jnp.zeros_like(
            x_mask[:, :1], dtype=bool
        )  # Use jnp.zeros_like for consistency
        # Concatenate along the first dimension with jnp.concatenate:
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
            # Concatenate masks along the first dimension with jnp.concatenate:
            x_mask = jnp.concatenate([prec_mask, x_mask], axis=1)
        return x, x_mask
