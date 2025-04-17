import pandas as pd
df = pd.read_csv('data.csv')
print(df)

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy= 'mean')
imp_array = imp_mean.fit_transform(df)
print(imp_array)

# this can be done w median
# simple input median in strategy

#using dropna drops the whole NA column

df.dropna()
