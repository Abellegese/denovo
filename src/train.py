"Fully Sharded Data Parellelism Based Training"
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
from utils import _render_hyperparams, create_dataframe, pickle_object
from jax import jit
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
import os
import warnings
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from flax.training.early_stopping import EarlyStopping
from rich import print

warnings.filterwarnings("ignore")
from rich.console import Console

console = Console()

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"
# Speed Up flags
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=false "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

PATH = "configs/base.yaml"


def _config(path):
    with open(path, "r") as conf:
        config = yaml.safe_load(conf)
    return config


# Reading the config
CONFIG = _config(PATH)
# Create a table
table = _render_hyperparams(CONFIG)
# Render the table
console.print(table)


def _build_encoder(config, train):
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
        train=train,
    )
    return model


@struct.dataclass
class ModelConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    output_dim: int = 128
    train: bool = True
    warmup_epochs: int = int(CONFIG["warmup_epochs"])
    num_epochs: int = int(CONFIG["epochs"])
    axis_name: str = "data"
    encoder: Any = _build_encoder(config=CONFIG, train=train)


def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    """
    Creates learning rate schedule.
    For more information about the cosine scheduler,
    check out the paper “SGDR: Stochastic Gradient Descent with Warm Restarts”.
    """
    warmup_fn = optax.linear_schedule(
        init_value=1e-7,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


def create_train_state(data):
    spectras, precursor, spectr_mask = data
    encoder = ModelConfig.encoder
    cpc = CPCModel(
        input_dim=ModelConfig.input_dim,
        hidden_dim=ModelConfig.hidden_dim,
        output_dim=ModelConfig.output_dim,
        batch_size=CONFIG["train_batch_size"],
        encoders=encoder,
        regressor=True,
        num_layers=CONFIG["n_layers"],
    )
    init_rngs = {
        "params": jax.random.key(0),
        "dropout": jax.random.key(1),
    }

    params = cpc.init(init_rngs, spectras, precursor, spectr_mask)["params"]
    schedule_fn = create_learning_rate_fn(
        ModelConfig, float(CONFIG["learning_rate"]), int(1.0 * 2_400_000)  # 15607: step per eopch
    )
    tx = optax.chain(
        optax.clip_by_global_norm(float(CONFIG["gradient_clip_val"])),  # gradient clipping
        optax.adamw(schedule_fn, b1=0.9, b2=0.98, eps=1e-9, weight_decay=1e-1),
    )
    return train_state.TrainState.create(apply_fn=cpc.apply, params=params, tx=tx)


def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def loss_fn(params, data, batch_size, train):
    spectra, precurs, spectr_mask = data
    ModelConfig.train = train
    # fold the dropout
    dropout_rng = fold_rng_over_axis(jax.random.key(1), ModelConfig.axis_name)

    loss, z, c = CPCModel(
        input_dim=ModelConfig.input_dim,
        hidden_dim=ModelConfig.hidden_dim,
        output_dim=ModelConfig.output_dim,
        batch_size=batch_size,
        encoders=ModelConfig.encoder,
        regressor=True,
        train=train,
        num_layers=CONFIG["n_layers"],
    ).apply(
        {"params": params},
        spectra,
        precurs,
        spectr_mask,
        rngs={"dropout": dropout_rng},
    )
    return loss

def evaluate_model(params, data, batch_size, train):
    return loss_fn(params, data, batch_size, train)

# Update apply_model function
def apply_model(state, data, batch_size, train, compute_grads=True):
    grad_fn = jax.value_and_grad(loss_fn)
    (loss), grads = grad_fn(state.params, data, batch_size, train)
    return grads, loss

def update_model(state, grads):
    return state.apply_gradients(grads=grads)

device_array = np.array(jax.devices())
mesh = Mesh(device_array, (ModelConfig.axis_name,))

console.print(
    f"[blue][INFO][/blue]:[green]device mesh has been created: [green]{mesh}[/green]"
)
# Shard and Compile Train State
init_dp_fn = jax.jit(
    shard_map(
        create_train_state,
        mesh,
        in_specs=(P(ModelConfig.axis_name)),
        out_specs=P(),
        check_rep=False,
    )
)

def train_step(state, batch):
    spectra, precursors, spectra_mask = batch
    grads, loss = apply_model(
        state,
        (spectra, precursors, spectra_mask),
        batch_size=CONFIG["train_batch_size"],
        train=True,
    )
    with jax.named_scope("sync_gradients"):
        grads = jax.tree_map(
            lambda g: jax.lax.pmean(g, 
            axis_name=ModelConfig.axis_name), 
            grads
        )
    with jax.named_scope("sync_metrics"):
        loss = jax.tree_map(
            lambda l: jax.lax.pmean(l, 
            axis_name=ModelConfig.axis_name), 
            loss
        )
    state = update_model(state, grads)
    return state, loss


def eval_step(batch, state):
    spectra, precursors, spectra_mask = batch
    val_loss = evaluate_model(
        state.params,
        (spectra, precursors, spectra_mask),
        CONFIG["train_batch_size"],
        train=False,
    )
    return val_loss


# Shard and Compile Training Steps
train_step_dp_fn = jax.jit(
    shard_map(
        train_step,
        mesh,
        in_specs=(P(), P(ModelConfig.axis_name)),
        out_specs=(P(), P()),
        check_rep=False,
    )
)

# Shard and Compile Validation Steps
val_step_dp_fn = jax.jit(
    shard_map(
        eval_step,
        mesh,
        in_specs=(P(ModelConfig.axis_name), P()),
        out_specs=(P()),
        check_rep=False,
    )
)
console.print("[blue][INFO][/blue]:[green]train and evaluate jitting completed[/green]")


def train(epochs, train_dataloader, val_dataloader, tds, vds, state, save=True):
    early_stop = EarlyStopping(min_delta=1e-3, patience=int(CONFIG["patient"]))
    epoch_train_loss, epoch_val_loss = [], []
    with mlflow.start_run():
        for epoch in range(epochs):
            for cnt, data in tqdm(
                enumerate(train_dataloader),
                total=(math.ceil(len(tds) / CONFIG["train_batch_size"])),
            ):
                spectra, precursors, spectra_mask, peptides, _, _ = data
                data = (
                    spectra.numpy(),
                    precursors.numpy(),
                    spectra_mask.unsqueeze(-1).numpy(),
                )
                state, loss = train_step_dp_fn(state, data)
                epoch_train_loss.append(loss)

            train_loss = np.mean(epoch_train_loss)

            for cnt, data in tqdm(
                enumerate(val_dataloader),
                total=(math.ceil(len(vds) / CONFIG["train_batch_size"])),
            ):
                spectra, precursors, spectra_mask, peptides, _, _ = data
                data = (
                    spectra.numpy(),
                    precursors.numpy(),
                    spectra_mask.unsqueeze(-1).numpy(),
                )

                eval_loss = val_step_dp_fn(data, state)
                epoch_val_loss.append(eval_loss)

            val_loss = np.mean(epoch_val_loss)
            # save the checkpoint in every 4 epoch
            print(
                f"[blue]Epoch: {epoch + 1}, train loss: {train_loss:.4f}, validation loss: {val_loss}[/blue]",
                flush=True,
            )
            # Early stopping
            early_stop = early_stop.update(val_loss)
            # Check if early stopping criteria are met
            if early_stop.should_stop:
                print(f"Met early stopping criteria, breaking at epoch {epoch}")
                break
            mlflow.log_metric(key="quality", value=2 * epoch, step=epoch)
            mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
            mlflow.log_metric("val_loss", value=val_loss, step=epoch)
    return state, epoch_train_loss, epoch_val_loss


def _save_model(path, state):
    # optimized serilizer
    ckpt = {"model": state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(path, ckpt, save_args=save_args)
    console.print(f"[blue][INFO][/blue]:[green]trained model daved at: {path}[/green]")

def _build_vocab(config):
    vocab = ["PAD", "<s>", "</s>"] + list(config["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    s2i = {v: k for k, v in i2s.items()}
    return vocab, s2i, i2s


@click.command()
@click.option("--learning_rate", default=5e-3, help="Learning rate for training")
@click.option("--momentum", default=0.9, help="Momentum for training")
@click.option(
    "--num_epochs", default=CONFIG["epochs"], help="Number of epochs for training"
)
@click.option(
    "--batch_size", default=CONFIG["train_batch_size"], help="Batch size for training"
)
@click.option("--save", default=True, help="Batch size for training")
@click.option("--save_path", help="to save the pretrained model")
def main(learning_rate, momentum, num_epochs, batch_size, save, save_path):
    # Load your data and create dataloade
    # dataset = load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="train")
    # val_dataset = load_dataset(
    #     "InstaDeepAI/ms_ninespecies_benchmark", split="validation"
    # )

    # building vocab
    vocab, s2i, i2s = _build_vocab(CONFIG)
    console.print("[blue][INFO][/blue]:[green]creating the dataframe[/green]")
    dataframe = create_dataframe()
    #dataframe = unpickle_object("train.pkl")
    # building dataset
    train_ds = SpectrumDataset(dataframe, s2i, CONFIG["n_peaks"], return_str=True)
    # val_ds = SpectrumDataset(val_dataset, s2i, CONFIG["n_peaks"], return_str=True)
    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    # print(f"length of the dataset {len(ds)}")
    # print(train_size, val_size)
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    console.print("[blue][INFO][/blue]:[green]dataset has been spilted for 90% train 10% validation[/green]")
    
    pickle_object(train_ds, "train.pkl")
    pickle_object(val_ds, "valid.pkl")
    console.print("[blue][INFO][/blue]:[green]dataset has been pickled[/green]")

    val_dataloader = DataLoader(
        val_ds, batch_size, shuffle=False, collate_fn=collate_batch, drop_last=True
    )
    train_dataloader = DataLoader(
        train_ds, batch_size, shuffle=True, collate_fn=collate_batch, drop_last=True
    )
    dataloader = (train_dataloader, val_dataloader)
    spectra, precursors, spectra_mask, peptides, _, _ = next(iter(train_dataloader))
    state = init_dp_fn(
        (spectra.numpy(), precursors.numpy(), spectra_mask.unsqueeze(-1).numpy())
    )
    console.print("[blue][INFO][/blue]:[green]model state has been crated.[/green]")
    count = sum(p.size for p in jax.tree_leaves(state.params))
    console.print(f"[blue][INFO][/blue]:[green]model size: {count/1_000_000:.1f}M[/green]")
    # Train the model
    state, train_loss, val_loss = train(
        num_epochs, train_dataloader, val_dataloader, train_ds, val_ds, state, save_path
    )
    _save_model(f"{CONFIG['save_path']}", state)
    console.print("[blue][INFO][/blue]:[green]model training Completed[/green]")

if __name__ == "__main__":
    main()
