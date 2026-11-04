import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE


data = np.loadtxt("pca_data.txt", delimiter=' ')
data -= np.mean(data, axis=0)  


cov_matrix = np.cov(data, rowvar=False)


eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]


top_eigenvectors = eigenvectors[:, :2]
projected_data = np.dot(data, top_eigenvectors)


umap_model = umap.UMAP()
umap_projection = umap_model.fit_transform(data)


tsne_model = TSNE(n_components=2, random_state=42)
tsne_projection = tsne_model.fit_transform(data)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA Plot
axes[0].scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
axes[0].set_title("PCA Projection")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

# UMAP Plot
axes[1].scatter(umap_projection[:, 0], umap_projection[:, 1], alpha=0.5)
axes[1].set_title("UMAP Projection")
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")

# t-SNE Plot
axes[2].scatter(tsne_projection[:, 0], tsne_projection[:, 1], alpha=0.5)
axes[2].set_title("t-SNE Projection")
axes[2].set_xlabel("t-SNE 1")
axes[2].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.show()