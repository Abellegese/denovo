import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import Any

@struct.dataclass
class Config:
  dtype: Any = jnp.bfloat16

class InfoNCELoss(nn.Module):
    gar_hidden: int
    genc_hidden: int
    pred_timestep: int
    batch_size: int
    negative_samples: int = 10

    @nn.compact
    def __call__(self, x, z, c):
        full_z = z
        Wc = self.predictor(c)
        return self.infonce_loss(Wc, z, full_z)

    def predictor(self, c):
        return nn.Dense(
            features=self.genc_hidden * self.pred_timestep,
            use_bias=False, dtype=Config.dtype
        )(c)

    def get_pos_sample_f(self, Wc_k, z_k):
        Wc_k = jnp.expand_dims(Wc_k, axis=1)
        z_k = jnp.expand_dims(z_k, axis=2)
        f_k = jnp.squeeze(jnp.matmul(Wc_k, z_k), 1)
        return f_k

    def get_neg_z(self, z):
        z = z.reshape(-1, z.shape[-1])
        z_neg = jnp.stack(
            [
                jnp.take(
                    z,
                    jax.random.permutation(jax.random.PRNGKey(0), z.shape[0]),
                    axis=0,
                )
                for _ in range(self.negative_samples)
            ],
            axis=2,
        )
        return z_neg, None, None

    def get_neg_samples_f(self, Wc_k, z_k, z_neg=None, k=None):
        # Wc_k = Wc_k.unsqueeze(1)
        Wc_k = jnp.expand_dims(Wc_k, axis=1)
        z_k_neg = z_neg[z_neg.shape[0] - Wc_k.shape[0] :, :, :]
        f_k = jnp.squeeze(jnp.matmul(Wc_k, z_k_neg), 1)
        return f_k

    def infonce_loss(self, Wc, z, full_z):
        seq_len, loss = z.shape[1], 0.0
        # sampling method 1 / 2
        z_neg, _, _ = self.get_neg_z(full_z)

        for k in range(1, self.negative_samples + 1):
            z_k = z[:, k:, :]
            Wc_k = Wc[
                :,
                :-k,
                (k - 1) * self.genc_hidden : k * self.genc_hidden,
            ]
            z_k = z_k.reshape(-1, z_k.shape[-1])
            Wc_k = Wc_k.reshape(-1, Wc_k.shape[-1])
            pos_samples = self.get_pos_sample_f(Wc_k, z_k)
            neg_samples = self.get_neg_samples_f(Wc_k, z_k, z_neg, k)

            # concatenate positive and negative samples
            results = jnp.concatenate((pos_samples, neg_samples), axis=1)
            _loss = nn.log_softmax(results)[:, 0]

            n_samples = (seq_len - k) * self.batch_size
            _loss = -_loss.sum() / n_samples
            loss += _loss
        # Normilizing the loss accross prediction timesteps
        loss /= self.pred_timestep
        return loss
