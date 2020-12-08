#!/usr/bin/env python3
"""
here some unnecessary documentation
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

plt.title('PCA of Iris Dataset')
plt.plasma()
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(pca_data[:, 0], pca_data[:, 1],
           pca_data[:, 2], c=labels)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.show()
