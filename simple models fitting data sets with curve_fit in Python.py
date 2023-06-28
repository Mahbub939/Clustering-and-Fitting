Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the model function
... def model_func(x, a, b, c):
...     # Example: low-order polynomial
...     return a + b * x + c * x**2
... 
... # Define input and output data
... xdata = np.array([1, 2, 3, 4, 5])
... ydata = np.array([3, 8, 19, 34, 52])
... 
... # Fit the data to the model
... popt, pcov = curve_fit(model_func, xdata, ydata)
... 
... # Calculate the confidence range using err_ranges function
... def err_ranges(xdata, ydata, popt, pcov, alpha=0.05):
...     popt = np.atleast_1d(popt)
...     if pcov is None:
...         # use leastsq covariance if None provided
...         pcov = np.atleast_2d(np.cov((ydata - model_func(xdata, *popt)), rowvar=False))
...     else:
...         pcov = np.atleast_2d(pcov)
... 
...     alpha = np.array([alpha/2, 1 - alpha/2])
...     q = np.array([stats.norm.ppf(a) for a in 1 - alpha])
...     err = np.sqrt(np.diag(pcov))
...     return popt + np.outer(q, err)
... 
... conf_int = err_ranges(xdata, ydata, popt, pcov)
... 
... # Generate data for prediction
... xpred = np.linspace(1, 10, 100)
... ypred = model_func(xpred, *popt)
... 
... # Plot the data and the best-fitting function with the confidence range
... plt.plot(xdata, ydata, 'ro', label='data')
... plt.plot(xpred, ypred, 'b-', label='best fit')
... plt.fill_between(xpred, model_func(xpred, *conf_int[0]), model_func(xpred, *conf_int[1]), alpha=0.2)
... plt.legend()
... plt.show()
