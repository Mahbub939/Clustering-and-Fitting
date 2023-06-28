Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... import numpy as np
... from sklearn.cluster import KMeans
... import matplotlib.pyplot as plt
... 
... # Load data from World Bank website
... df = pd.read_csv('data.csv', index_col=0)
... 
... # Normalize the data
... df_norm = (df - df.mean()) / df.std()
... 
... # Apply K-means clustering
... kmeans = KMeans(n_clusters=3)
... kmeans.fit(df_norm)
... labels = kmeans.labels_
... centroids = kmeans.cluster_centers_
... 
... # Add cluster membership as a new column
... df['Cluster'] = labels
... 
... # Display original values and cluster membership
... print(df)
... 
... # Generate scatter plot
... plt.scatter(df_norm['CO2 emissions per capita'], df_norm['GDP per capita'], c=labels)
... plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='r')
... plt.xlabel('CO2 emissions per capita')
... plt.ylabel('GDP per capita')
... plt.show()
