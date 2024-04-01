import orbax
from flax.training import orbax_utils
from model import CPCModel
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import time
import mlflow
import time
import optax
import orbax
import yaml
from utils import *
from dataset import SpectrumDataset, collate_batch
from datasets import load_dataset
from model import CPCModel
from layers import Encoder
from loss import InfoNCELoss
from flax import jax_utils
from flax.training import train_state, checkpoints
from tqdm.auto import tqdm
from flax.training import orbax_utils
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from flax.training import checkpoints
from flax import struct
from typing import Any
from jax import lax
import json
from flax.training.early_stopping import EarlyStopping
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_d


PATH = 'configs/base.yaml'
def _config(path):
    with open(path, "r") as conf:
        config = yaml.safe_load(conf)
    return config

# Reading the config
CONFIG = _config(PATH)
def _count_params(params): return sum(p.size for p in jax.tree_leaves(params))
def _build_vocab(config):
    vocab = ["PAD", "<s>", "</s>"] + list(config["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    s2i = {v: k for k, v in i2s.items()}
    return vocab, s2i, i2s
def _build_encoder(config):
    model = Encoder(
        residues=config["residues"],
        dim_model=512,
        n_head=config["n_head"],
        dim_feedforward=512,
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        max_charge=config["max_charge"],
        use_depthcharge=config["use_depthcharge"],
        dec_precursor_sos=config["dec_precursor_sos"],
        train=False
    )
    return model
encoder = _build_encoder(CONFIG)
def load_and_use_encoder(spectra, precursor=None, spectr_mask=None):

  # Load the model state from the checkpoint
  # state = _load_model(checkpoint_path)
  model = CPCModel(
    input_dim = 512,
    hidden_dim = 256,
    output_dim = 256,
    encoders=encoder,
    batch_size=1,
    regressor=True,
  )
  init_rngs = {
        "params": jax.random.key(0),
        "dropout": jax.random.key(1),
  }
  params = model.init(init_rngs, spectra, precursor, spectr_mask)

  # Extract model parameters from the loaded state
  empty_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=jax.tree_map(np.zeros_like, params),  # values of the tree leaf doesn't matter
    tx=optax.adam(5e-3)
,
   )
  target = {'model': empty_state}
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  
  state = orbax_checkpointer.restore('/home/abellegese/Videos/pipeline/tests/', item=target)
  # print(state)
  # params = state['model']['params']
  # print(params)
  # Create a CPCModel instance
  file_path = "state.json"
  # state_json_string = json.dumps(state, indent=4)
  # Save 'state' as JSON
  with open(file_path, 'w') as text_file:
      text_file.write(str(state))

  # Initialize the model with loaded parameters
  z, c = model.apply(state, spectra, precursor, spectr_mask, method=model.get_latent_representations)

  # Use the encoder function to get latent representations
#   z, c = model.get_latent_representations(spectra, precursor, spectr_mask)

  return z, c

print("Dataset loaded")
dataset = load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test[:1%]")
    # building vocab
vocab, s2i, i2s = _build_vocab(CONFIG)
# taking 5% percent of it just for experiment
ds = SpectrumDataset(dataset, s2i, CONFIG["n_peaks"], return_str=True)

train_size = int(0.85 * len(ds))
val_size = len(ds) - train_size
print(f'length of the dataset {len(ds)}')
print(train_size, val_size)
train_dataset, val_dataset = random_split(
    ds, [train_size, val_size]
)

del val_dataset
train_dataloader = DataLoader(
    train_dataset, 1, shuffle=True, collate_fn=collate_batch
)
print("Dataset loader created")

spectra, precursors, spectra_mask, peptides, _ = next(iter(train_dataloader))
print("subset created")

z, c = load_and_use_encoder(spectra, precursors, spectra_mask.unsqueeze(-1).numpy())
print("Done!!!")
