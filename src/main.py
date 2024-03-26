import orbax
from flax.training import orbax_utils

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


raw_restored = orbax_checkpointer.restore("/home/abellegese/Videos/pipeline/artifacts/")
print(raw_restored)
