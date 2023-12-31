import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load data and labels
data = np.load('data.npy')
labels = np.load('labels.npy')

# Perform PCA for dimensionality reduction to 2D and 3D
pca_2d = PCA(n_components=2).fit(data)
pca_3d = PCA(n_components=3).fit(data)
data_2d = pca_2d.fit_transform(data)
data_3d = pca_3d.fit_transform(data)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, edgecolor='black')
plt.title('PCA - 2D')

ax = plt.subplot(2, 1, 2, projection='3d')
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, edgecolor='black')
ax.set_title('PCA - 3D')
plt.show()

print("We can see that the 3D representation allow a better separation between the classes,\nwhere in 2D there's some overlaps.")