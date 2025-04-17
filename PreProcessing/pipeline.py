import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer

df = pd.read_csv("drawndata2.csv")
X = df[['x' , 'y']].values
y = df['z'] == 'a'
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()

# so for this what we will need to do is to seperate the colors in to two parts 
# this is done using pipeline 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", QuantileTransformer(n_quantiles=100)),
    ("model", LogisticRegression())
])

pred = pipe.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c = pred)

# to improve the graph we will import polomial features

pipe2 = Pipeline([
    ("scale", PolynomialFeatures()),
    ("model", LogisticRegression())
])
pred = pipe2.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c = pred)
plt.show()
