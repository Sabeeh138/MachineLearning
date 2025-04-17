import pandas as pd
df = pd.read_csv("Iris.csv")
df.head()
print(df.head())
print(df.tail())

species = df["Species"].unique()
print(species)
val = df["Species"].value_counts()
print(val)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
print(labelencoder)

df["Species"] = labelencoder.fit_transform(df["Species"])
print(df.head())
Dtypes = df.dtypes
print(Dtypes)

Uni = df["Species"].unique()
print(Uni)

print(df["Species"].value_counts())