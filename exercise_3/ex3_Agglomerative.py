import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


data = np.load('data.npy')

agglomerative = AgglomerativeClustering(n_clusters=3)
labels_agglomerative = agglomerative.fit_predict(data)

# Evaluate clustering using silhouette score
silhouette_agglomerative = []

for k in range(2, 11):
    agglomerative = AgglomerativeClustering(n_clusters=k)
    labels_agglomerative = agglomerative.fit_predict(data)
    silhouette_agglomerative.append(silhouette_score(data, labels_agglomerative))

plt.plot(range(2, 11), silhouette_agglomerative, marker='o')
plt.title('Silhouette Score for Agglomerative Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

print("We can see that the optimal number of clusters is 6,\nsince after that the silhouette score decrease.\nSilhouette score is a measure of cluster cohesion and separation, so the higher the better.")