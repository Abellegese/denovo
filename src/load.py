import jax.numpy as jnp
import flax.linen as nn
import jax
from flax.training import train_state, checkpoints
from tqdm.auto import tqdm
from flax.training import orbax_utils
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from flax.training import checkpoints
from flax import struct
from typing import Any
import optax
import orbax
from utils import *
def _save_model(path, ckpt):
    # optimized serilizer
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(path, ckpt, save_args=save_args)

# Define a model class
class MyModel(nn.Module):
    def setup(self):
        # Define a parameter
        self.weight = self.param('weight', lambda rng_key, shape: jnp.zeros(shape), (3, 3))  # Example parameter initialized with zeros

    def __call__(self, x):
        # Use the parameter in the forward pass
        output = jnp.matmul(x, self.weight)
        return output
# x1 = jnp.zeros((3, 3))
# # Create an instance of the model
# model = MyModel()
# variables = model.init(jax.random.PRNGKey(0), x1)
# tx = optax.sgd(learning_rate=0.001)      # An Optax SGD optimizer.
# state = train_state.TrainState.create(
#     apply_fn=model.apply,
#     params=variables['params'],
#     tx=tx)
# # Perform a simple gradient update similar to the one during a normal training workflow.
# state = state.apply_gradients(grads=jax.tree_map(jnp.ones_like, state.params))

# # Some arbitrary nested pytree with a dictionary and a NumPy array.

# # Bundle everything together.
# ckpt = {'model': state, 'data': [x1]}
# _save_model("/home/abellegese/Videos/pipeline/artifacts/", ckpt)
def _load_model(path):
  """Loads a previously saved Flax model state from the given path.

  Args:
      path: The file path where the model checkpoint is stored.

  Returns:
      A Flax TrainState object containing the loaded model parameters and optimizer state.
  """
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

  raw_restored = orbax_checkpointer.restore(path)

  return raw_restored
variables = _load_model("/home/abellegese/Videos/pipeline/artifacts/")

# Access the parameter
# print(variables['model']['params']['weight'])
print(variables)
