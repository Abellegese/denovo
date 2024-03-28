import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import math
import numpy as np
import click
import time
import mlflow
import time
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

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
jax.config.update("jax_platform_name", "gpu")
# Config Path
PATH = 'configs/base.yaml'
def _config(path):
    with open(path, "r") as conf:
        config = yaml.safe_load(conf)
    return config

# Reading the config
CONFIG = _config(PATH)


def _build_encoder(config):
    model = Encoder(
        residues=config["residues"],
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        max_charge=config["max_charge"],
        use_depthcharge=config["use_depthcharge"],
        dec_precursor_sos=config["dec_precursor_sos"],
    )
    return model


@struct.dataclass
class ModelConfig:
    input_dim: int = 768
    hidden_dim: int = 166
    output_dim: int = 166
    encoder: Any = _build_encoder(config=CONFIG)


# @functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_train_state(
    rng, learning_rate, momentum, spectras, precursor, spectr_mask, batch_size
):
    encoder = ModelConfig.encoder
    cpc = CPCModel(
        input_dim=ModelConfig.input_dim,
        hidden_dim=ModelConfig.hidden_dim,
        output_dim=ModelConfig.output_dim,
        batch_size=batch_size,
        encoders=encoder,
        regressor=True
    )
    init_rngs = {
        "params": jax.random.key(0),
        "dropout": jax.random.key(1),
    }
    params = cpc.init(init_rngs, spectras, precursor, spectr_mask)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=cpc.apply, params=params, tx=tx)

def loss_fn(params, data, batch_size):
    spectra, precurs, spectr_mask = data
    spectr_mask = spectr_mask.unsqueeze(2).numpy()

    loss, z, c = CPCModel(
        input_dim=ModelConfig.input_dim,
        hidden_dim=ModelConfig.hidden_dim,
        output_dim=ModelConfig.output_dim,
        batch_size=batch_size,
        encoders=ModelConfig.encoder,
        regressor=True
    ).apply(
        {"params": params},
        spectra,
        precurs,
        spectr_mask,
        rngs={"dropout": jax.random.key(5)},
    )
    return loss

def evaluate_model(params, data, batch_size): return loss_fn(params, data, batch_size)

# Update apply_model function
def apply_model(state, data, batch_size, compute_grads=True):
    # spectra, precurs, spectr_mask = data
    # spectr_mask = spectr_mask.unsqueeze(2).numpy()
    # Normal Loss
    grad_fn = jax.value_and_grad(loss_fn)
    (loss), grads = grad_fn(state.params, data, batch_size)
    return grads, loss

# @jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_one_epoch(state, dataloader, num_epochs, size, batch_size):
    epoch_train_loss, epoch_val_loss = [], []
    train_dataloader, val_dataloader = dataloader

    with mlflow.start_run():
        for epoch in range(num_epochs):
            for cnt, data in tqdm(
                enumerate(train_dataloader), total=(math.ceil(size / batch_size))
            ):
                spectra, precursors, spectra_mask, peptides, _ = data
                grads, loss = apply_model(
                    state,
                    (spectra, precursors, spectra_mask),
                    batch_size=batch_size,
                )
                
                del spectra, precursors, spectra_mask, peptides, _, data
                state = update_model(state, grads)
                del grads
            # epoch_loss.append(jax_utils.unreplicate(loss))
            epoch_train_loss.append(loss)
            train_loss = np.mean(epoch_train_loss)

            for cnt, data in tqdm(
                enumerate(val_dataloader), total=(math.ceil(size / batch_size))
            ):
                spectra, precursors, spectra_mask, peptides, _ = data
                val_loss = evaluate_model(
                    state.params,
                    (spectra, precursors, spectra_mask),
                    batch_size
                )
                del spectra, precursors, spectra_mask, peptides, _, data
                # state = update_model(state, grads)
            # epoch_loss.append(jax_utils.unreplicate(loss))
            epoch_val_loss.append(val_loss)
            val_loss = np.mean(epoch_val_loss)         

            print(
                f"Epoch: {epoch + 1}, train loss: {train_loss:.4f}, validation loss: {val_loss}",
                flush=True,
            )
            # Mlflow logging
            now = time.time()
            mlflow.log_metric(key="quality", value=2 * epoch, step=epoch)
            mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
            mlflow.log_metric("val_loss", value=val_loss, step=epoch)

    return state, epoch_train_loss


def _save_model(path, ckpt):
    # optimized serilizer
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(path, ckpt, save_args=save_args)


def _build_vocab(config):
    vocab = ["PAD", "<s>", "</s>"] + list(config["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    s2i = {v: k for k, v in i2s.items()}
    return vocab, s2i, i2s


@click.command()
@click.option("--learning_rate", default=1e-5, help="Learning rate for training")
@click.option("--momentum", default=0.9, help="Momentum for training")
@click.option("--num_epochs", default=10, help="Number of epochs for training")
@click.option("--batch_size", default=32, help="Batch size for training")
@click.option("--save", default=True, help="Batch size for training")
def main(learning_rate, momentum, num_epochs, batch_size, save):
    # Load your data and create dataloade
    dataset = load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test[:1%]")
    # building vocab
    vocab, s2i, i2s = _build_vocab(CONFIG)
    # taking 5% percent of it just for experiment
    ds = SpectrumDataset(dataset, s2i, CONFIG["n_peaks"], return_str=True)
    # train_size = int(0.85 * len(ds))
    # test_size = len(ds) - train_size
    # train_dataset, test_dataset = random_split(ds, [train_size, test_size])
    # val_dataloader = DataLoader(
    #     test_dataset, batch_size, shuffle=False, collate_fn=collate_batch
    # )
    # train_dataloader = DataLoader(
    #     test_dataset, batch_size, shuffle=True, collate_fn=collate_batch
    # )
    train_size = int(0.05 * len(ds))  # 10% of the data for training
    test_size = int(0.01 * len(ds))   # 1% of the data for testing
    val_size = len(ds) - train_size - test_size
    print(train_size, test_size, val_size)
    train_dataset, test_dataset, val_dataset = random_split(ds, [train_size, test_size, val_size])
    # val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_batch)
    del val_dataset
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_batch)

    dataloader = (train_dataloader, val_dataloader)
    spectra, precursors, spectra_mask, peptides, _ = next(iter(train_dataloader))
    # Initialize the training state
    rng = jax.random.PRNGKey(4)
    state = create_train_state(
        rng,
        learning_rate,
        momentum,
        spectra,
        precursors,
        spectra_mask.unsqueeze(2).numpy(),
        batch_size
    )
    # state = jax_utils.replicate(state)
    start = time.time()
    state, epoch_loss = train_one_epoch(
        state, dataloader, num_epochs, len(test_dataset), batch_size
    )
    # creat a checkpoint
    ckpt = {"model": state}
    # save the model
    if save:
        _save_model("/home/abellegese/Videos/pipeline/artifacts/", ckpt)

    print("Total time: ", time.time() - start, "seconds", "Loss", epoch_loss)

if __name__ == "__main__":
    main()
