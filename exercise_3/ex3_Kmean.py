import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



data = np.load('data.npy')

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels_kmeans = kmeans.fit_predict(data)


print("K-Means metric (Inertia)", kmeans.inertia_)

#heuristics part 

inertia_values = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(data)
    inertia_values.append(kmeans.inertia_)

    
# Plot the Elbow curve
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

print("The Elbow Method shows that the optimal number of clusters is 6, since after that the inertia doesn't seem to decreases.\nInertia is the cluster compactness measure, so the lower the better.")