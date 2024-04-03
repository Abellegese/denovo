import numpy as np
from sklearn.manifold import TSNE

def normalize_embeddings(embeddings):
    """
    Normalize embeddings along the last dimension.
    
    Args:
        embeddings (numpy.ndarray): Array of shape (num_samples, embedding_length).
        
    Returns:
        numpy.ndarray: Normalized embeddings.
    """
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

def compute_tsne_embeddings(embeddings, n_components=2, **kwargs):
    """
    Compute t-SNE embeddings of normalized embeddings.
    
    Args:
        embeddings (numpy.ndarray): Array of shape (num_samples, embedding_length).
        n_components (int): Number of components for t-SNE.
        **kwargs: Additional arguments to pass to sklearn.manifold.TSNE.
        
    Returns:
        numpy.ndarray: t-SNE embeddings.
    """
    normalized_embeddings = normalize_embeddings(embeddings)
    tsne = TSNE(n_components=n_components, **kwargs)
    tsne_embeddings = tsne.fit_transform(normalized_embeddings)
    return tsne_embeddings

# Assuming 'embeddings' is your tensor of shape (5, 79, 768)
# Reshape to treat each embedding vector independently
reshaped_embeddings = embeddings.reshape((-1, 768))

# Compute t-SNE embeddings
tsne_embeddings = compute_tsne_embeddings(reshaped_embeddings)

# Plot the embeddings
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=10)
plt.title('t-SNE Visualization of Transformer Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
