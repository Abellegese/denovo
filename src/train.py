import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import math
import numpy as np
import click
import time
import yaml
import mlflow
import time
import orbax
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
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

file_path = "configs/base.yaml"
with open(file_path, "r") as conf:
    config = yaml.safe_load(conf)




def _build_encoder():
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
        dec_precursor_sos=False,
    )
    return model


# @functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_train_state(
    rng, learning_rate, momentum, spectras, precursor, spectr_mask, batch_size
):
    """Creates initial `TrainState`."""
    encoder = _build_encoder()
    cpc = CPCModel(input_dim=768, hidden_dim=166,
                   output_dim=166, batch_size=batch_size, encoders=encoder)
    init_rngs = {
        "params": jax.random.key(0),
        "dropout": jax.random.key(1),
    }
    params = cpc.init(init_rngs, spectras, precursor, spectr_mask)["params"]
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=cpc.apply, params=params, tx=tx)


# @functools.partial(jax.pmap, axis_name="ensemble")
def apply_model(state, data, input_dim, hidden_dim,
                output_dim, batch_size):
    spectra, precurs, spectr_mask = data
    spectr_mask = spectr_mask.unsqueeze(2).numpy()
    encoder = _build_encoder()

    def loss_fn(params):
        loss, z, c = CPCModel(input_dim=768, hidden_dim=166,
                              output_dim=166, batch_size=batch_size, encoders=encoder).apply(
            {"params": params},
            spectra,
            precurs,
            spectr_mask,
            rngs={"dropout": jax.random.key(5)},
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    (loss), grads = grad_fn(state.params)
    return grads, loss


# @jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

def train_one_epoch(state, dataloader, num_epochs, size, batch_size):
    """Train for 1 epoch on the training set."""
    epoch_loss = []
    with mlflow.start_run():
        for epoch in range(num_epochs):
            for cnt, data in tqdm(enumerate(dataloader), total=(math.ceil(size / batch_size))):
                spectra, precursors, spectra_mask, peptides, _ = data
                grads, loss = apply_model(
                    state, (spectra, precursors,
                            spectra_mask),  input_dim=768, hidden_dim=166,
                    output_dim=166, batch_size=batch_size)
                state = update_model(state, grads)
            # epoch_loss.append(jax_utils.unreplicate(loss))
            epoch_loss.append(loss)
            train_loss = np.mean(epoch_loss)

            print(
                f"Epoch: {epoch + 1}, train loss: {train_loss:.4f}",
                flush=True,
            )
            # Mlflow logging
            now = time.time()
            mlflow.log_metric(key="quality", value=2 * epoch, step=epoch)
            mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

    return state, epoch_loss

# def _save_model(path, ckpt):
#     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     save_args = orbax_utils.save_args_from_target(ckpt)
#     orbax_checkpointer.save(path, ckpt, save_args=save_args)

def _save_model(path, ckpt):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(path, ckpt, save_args=save_args)

@click.command()
@click.option("--learning_rate", default=1e-3, help="Learning rate for training")
@click.option("--momentum", default=0.9, help="Momentum for training")
@click.option("--num_epochs", default=10, help="Number of epochs for training")
@click.option("--batch_size", default=32, help="Batch size for training")

def main(learning_rate, momentum, num_epochs, batch_size):
    # Load your data and create dataloade
    dataset = load_dataset(
        "InstaDeepAI/ms_ninespecies_benchmark", split="test[:1%]")

    vocab = ["PAD", "<s>", "</s>"] + list(config["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    # next(iter(dataloader))

    s2i = {v: k for k, v in i2s.items()}
    # taking 2% percent of it
    ds = SpectrumDataset(dataset, s2i, config["n_peaks"], return_str=True)
    train_size = int(0.95 * len(ds))
    test_size = len(ds) - train_size
    train_dataset, test_dataset = random_split(ds, [train_size, test_size])
    dataloader = DataLoader(
        test_dataset,
        batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    spectra, precursors, spectra_mask, peptides, _ = next(iter(dataloader))
    # Initialize the training state
    rng = jax.random.PRNGKey(4)
    state = create_train_state(
        rng, learning_rate, momentum, spectra, precursors, spectra_mask.unsqueeze(
            2).numpy(), batch_size
    )
    # state = jax_utils.replicate(state)
    start = time.time()
    state, epoch_loss = train_one_epoch(
        state, dataloader, num_epochs, len(test_dataset), batch_size)
    # creat a checkpoint
    ckpt = {'model': state}
    # save the model
    _save_model('/home/abellegese/Videos/pipeline/artifacts/', ckpt)

    print("Total time: ", time.time() - start, "seconds", "Loss", epoch_loss)


if __name__ == "__main__":
    main()
