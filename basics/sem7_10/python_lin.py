# Fit linear and quadratic surfaces to data
# Based on https://github.com/probml/pmtk3/blob/master/demos/surfaceFitDemo.m


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress


#url = "https://raw.githubusercontent.com/probml/probml-data/main/data/moteData/moteData.mat"
#response = requests.get(url)
# rawdata = response.text
#rawdata = BytesIO(response.content)
data = loadmat(".\\basics\\sem7_10\\DataSurfFit") # грузит матлабовский файл


X = data["X"] # initial two-column data
y = data["y"].flatten() # dependent variable column

n = len(y)
X_pad = np.column_stack((np.ones(n), X)) 

phi = X_pad # predictors matrix including dummy variable
fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d') #first image 
ax.set_zlim(15, 19)
ax.scatter(X[:, 0], X[:, 1], y, color="r")
n = 10
xrange = np.linspace(min(X[:, 0]), max(X[:, 0]), n)
yrange = np.linspace(min(X[:, 1]), max(X[:, 1]), n)
xx, yy = np.meshgrid(xrange, yrange)
flatxx = xx.reshape((n**2, 1))
flatyy = yy.reshape((n**2, 1))
w = np.linalg.lstsq(phi, y)[0]
z = np.column_stack((flatxx, flatyy))
z = np.column_stack((np.ones(n**2), z))
f = np.dot(z, w)
ax.plot_surface(xx, yy, f.reshape(n, n), rstride=1, cstride=1, cmap="jet")


ax = fig.add_subplot(1,2,2, projection='3d') #second image 
ax.set_zlim(15, 19)
ax.scatter(X[:, 0], X[:, 1], y, color="r")
phi = np.column_stack((X_pad, X**2))
w = np.linalg.lstsq(phi, y)[0]
z = np.column_stack((z, flatxx**2, flatyy**2))
f = np.dot(z, w)
ax.plot_surface(xx, yy, f.reshape(n, n), rstride=1, cstride=1, cmap="jet")
plt.show()
