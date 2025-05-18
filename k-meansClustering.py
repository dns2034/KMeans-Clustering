import numpy as np  # for numerical operations
import matplotlib.pyplot as plt   # for creating visualizations, such as plots and charts
from sklearn.cluster import KMeans  #algorithm for clustering data into groups
from sklearn.datasets import make_blobs  # used to generate synthetic datasets for clustering

# Generates a dataset with 300 samples (data points) that are grouped into 4 centers (clusters)
# 'X' contains the coordinates of the data points
# 'y' containes the true labels (which cluster each point belongs to)
# The 'random_state' parameter ensures  that the results are reproducible
X, y = make_blobs(n_samples=300, centers=4, random_state=42)


kmeans = KMeans(n_clusters=4, random_state=42)  # creates an instance of the KMeans algorithm
kmeans.fit(X)  # finds the centroids of the clusters and assigns each data point to the nearest centroids
labels = kmeans.labels_  # retrieves the labels (cluster assignments) for each data point after fittinh the model
centroids = kmeans.cluster_centers_  # retrieves the coordinates of the centroids (center points of each cluster) after fitting the model

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')  # creates a scatter plot of the data points in 'X'. ('c=labels' argument colors the poiints based on their cluster assignments, and 'cmap='viridis' specifies the color map used for the points)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')  # adds another scatter plot for the centroids, marking then with a red 'X'. ('s=200' = size of the markers, and 'label='Centroids' = label for the legend)
plt.title("K-Means Clustering Example") # set the title of the plot to "K-Means CLustering Example"
plt.legend()  # displays the legend of the plot
plt.show()  # displays the plot on the screen