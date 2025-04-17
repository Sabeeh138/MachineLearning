import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer

arr = np.array(["low", "low", "low", "medium"]).reshape(-1,1)
print(arr)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False, handle_unknown= 'ignore')
enc.fit_transform(arr)
print(arr)