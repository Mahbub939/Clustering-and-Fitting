Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load data for clustering
data = pd.read_csv('countries.csv')
attributes = ['GDP per capita', 'CO2 emissions per capita', 'Renewable energy consumption']
X = data[attributes].values

# normalize data for clustering
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# apply K-means clustering to identify clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_norm)
data['Cluster'] = kmeans.labels_

# select one country from each cluster
countries = []
for cluster in range(3):
    country = data[data['Cluster'] == cluster].sample(1)['Country'].iloc[0]
    countries.append(country)

# compare selected countries within and across clusters
... for i in range(len(countries)):
...     for j in range(i+1, len(countries)):
...         country1 = countries[i]
...         country2 = countries[j]
...         data1 = data[data['Country'] == country1][attributes].values[0]
...         data2 = data[data['Country'] == country2][attributes].values[0]
...         print(f"Comparison between {country1} and {country2}:")
...         for k in range(len(attributes)):
...             attr = attributes[k]
...             val1 = data1[k]
...             val2 = data2[k]
...             diff = np.abs(val1 - val2)
...             print(f"{attr}: {val1} vs. {val2} (difference: {diff:.2f})")
...         print()
... 
... # fit a low-order polynomial function to CO2 emissions over time
... yearly_data = pd.read_csv('yearly_emissions.csv')
... x = yearly_data['Year'].values
... y = yearly_data['CO2 emissions'].values
... 
... def model_func(x, a, b, c):
...     return a * x**2 + b * x + c
... 
... params, cov = curve_fit(model_func, x, y)
... 
... # generate predictions and confidence intervals
... x_pred = np.arange(2020, 2051)
... y_pred = model_func(x_pred, *params)
... y_err = err_ranges(x_pred, x, y, params, cov)
... 
... # plot the data, best-fitting function, and confidence interval
... fig, ax = plt.subplots()
... ax.plot(x, y, 'o', label='data')
... ax.plot(x_pred, y_pred, '-', label='fit')
... ax.fill_between(x_pred, y_pred - y_err, y_pred + y_err, alpha=0.3)
... ax.set_xlabel('Year')
... ax.set_ylabel('CO2 emissions (metric tons per capita)')
... ax.legend()
... plt.show()
