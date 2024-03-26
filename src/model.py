from loss import InfoNCELoss
import jax
import jax.numpy as jnp
import flax.linen as nn


class AutoregressiveModel(nn.Module):
    output_dim: int
    n_heads: int

    def setup(self):
        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.output_dim
        )
        self.layer_norm = nn.LayerNorm()
        self.fc = nn.Dense(self.output_dim)

    @nn.compact
    def __call__(self, x):
        out = self.mha(x)  # Self attention
        out = self.layer_norm(x + out)  # Add & Norm
        out = self.fc(out)  # Feed forward
        return out


class CPCModel(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    batch_size: int
    encoders: any

    def setup(self):
        self.encoder = self.encoders
        self.autoregressor = AutoregressiveModel(self.hidden_dim, self.output_dim)
        self.loss = InfoNCELoss(
            self.hidden_dim,
            self.input_dim,
            batch_size=self.batch_size,
            pred_timestep=12,
        )

    def get_latent_size(self, input_size):
        x = jnp.zeros(input_size)
        z, c = self.get_latent_representations(x, x, x)
        return c.shape[-2], c.shape[-1]

    def get_latent_representations(self, spectra, precurs, spectr_mask):
        z = self.encoder(spectra, precurs, spectr_mask)
        z = z[0]
        c = self.autoregressor(z)
        return z, c

    def __call__(self, spectra, precurs, spectr_mask):
        z, c = self.get_latent_representations(spectra, precurs, spectr_mask)
        loss = self.loss(spectra, z, c)
        return loss, z, c
