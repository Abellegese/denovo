import jax.numpy as jnp
import math
import flax.linen as nn
import jax
from rich.table import Table
import pandas as pd
import polars as pl
from rich.console import Console
console = Console()
import pickle
import json

with open('configs/links.json', 'r') as f:
    links = json.load(f)

columns = [
    "experiment_name",
    "evidence_index",
    "scan_number",
    "sequence",
    "modified_sequence",
    "precursor_mass",
    "precursor_mz",
    "precursor_charge",
    "retention_time",
    "mz_array",
    "intensity_array",
    "title",
    "scans",
]


def expand_mask(mask):
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, axis=1)
    elif mask.ndim == 2:
        mask = jnp.expand_dims(jnp.expand_dims(mask, axis=0), axis=1)
    return mask


def _get_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.triu(jnp.ones((seq_len, seq_len))) == 1
    mask = mask.astype(jnp.float32)
    mask = mask.at[mask == 0].set(-jnp.inf)
    mask = mask.at[mask == 1].set(0.0)
    return jnp.transpose(mask)


def _count_params(params):
    return sum(p.size for p in jax.tree_leaves(params))


def _build_vocab(config):
    vocab = ["PAD", "<s>", "</s>"] + list(config["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    s2i = {v: k for k, v in i2s.items()}
    return vocab, s2i, i2s


def _render_hyperparams(CONFIG, n_devices):
    table = Table(title="Hyperparameters")
    table.add_column("Hyperparameter", style="cyan", justify="center")
    table.add_column("Value", style="magenta", justify="center")

    table.add_row("Number of Heads (n_head)", str(CONFIG["n_head"]))
    table.add_row("Number of Layers (n_layers)", str(CONFIG["n_layers"]))
    table.add_row("Feed Forward Dimension", str(CONFIG["dim_feedforward"]))
    table.add_row("Model Dimension", str(CONFIG["dim_model"]))
    table.add_row("Dropout", str(CONFIG["dropout"]))
    table.add_row("Max Sequence Length (max_length)", str(CONFIG["max_length"]))
    table.add_row("Learning rate", str(CONFIG["learning_rate"]))
    table.add_row("Warm up epochs", str(CONFIG["warmup_epochs"]))
    table.add_row("Patient", str(CONFIG["patient"]))
    table.add_row(
        "Batch per device", str(int(CONFIG["train_batch_size"]) / int(len(n_devices)))
    )  # assuming 2 devices
    table.add_row("Epochs", str(CONFIG["epochs"]))
    return table


def create_validation(df):
    sz = int(len(df) * 0.1)
    valid = df.sample(n=sz, random_state=42)
    df = df.drop(valid.index)
    return df, valid


def create_dataframe(N=220_000):
    df = pd.DataFrame(
        pl.read_ipc(links["apis_mellifera"]), 
        columns=columns
    )
    df1 = pd.DataFrame(
        pl.read_ipc(links["bacillus_subtilis"]), 
        columns=columns
    )
    df4 = pd.DataFrame(
        pl.read_ipc(links["methanosarcina_mazei"]), 
        columns=columns
    )
    df6 = pd.DataFrame(
        pl.read_ipc(links["saccharomyces_cerevisiae"]),
        columns=columns,
    )
    df7 = pd.DataFrame(
        pl.read_ipc(links["solanum_lycopersicum"]), 
        columns=columns
    )
    # df8 = pd.DataFrame(pl.read_ipc(download_links["vigna_mungo"]), columns=columns)
    console.print("[blue][INFO][/blue]:[green]dataframe has been downloaded[/green]")

    # df, df1, df4, df6, df7 = df.sample(N),df1.sample(N), df4.sample(N), df6.sample(N), df7.sample(N)
    console.print("[blue][INFO][/blue]:[green]dataframe has been sampled[/green]")
    
    (df, vl1), (df1, vl2), (df4, vl3), (df6, vl4), (df7, vl5), (df8, v6) = (
        create_validation(df),
        create_validation(df1),
        create_validation(df4),
        create_validation(df6),
        create_validation(df7),
        create_validation(df8),

    )
    train = pd.concat([df, df1, df4, df6, df7, df8])
    validation = pd.concat([vl1, vl2, vl3, vl4, vl5, vl8])
    console.print("[blue][INFO][/blue]:[green]train and valid dataframe created[/green]")

    train.to_pickle("/home/abellegese_aims_ac_za/denovo/train.pkl")
    validation.to_pickle("/home/abellegese_aims_ac_za/denovo/valid.pkl")
    # console.print("[blue][INFO][/blue]:[green]dataframe has been pickled[/green]")

    return (
        train,
        validation,
        {"size": [len(df), len(df1), len(df4), len(df6), len(df7)]},
    )

def pickle_object(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def unpickle_object(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj
