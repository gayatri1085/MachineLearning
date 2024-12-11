
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("customer_data.csv")

X = data[['feature1', 'feature2']]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

data['Cluster'] = kmeans.labels_

plt.scatter(data['feature1'], data['feature2'], c=data['Cluster'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()