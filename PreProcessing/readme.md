# 🧪 Machine Learning Preprocessing — Complete Summary

This README documents all the key preprocessing techniques I've studied and practiced as part of my machine learning journey. These are essential steps applied **before feeding data into a model**, ensuring it's clean, consistent, and meaningful for training.

---

## 🔧 1. Handling Missing Values
> Real-world data is messy — missing values are common. These techniques help deal with them effectively.

- **SimpleImputer**
  - `strategy='mean'` — replaces missing values with the mean (for numerical data)
  - `strategy='median'` — replaces with median (good for skewed data)
  - `strategy='most_frequent'` — replaces with the most common value (for categorical data)
  - `strategy='constant'` — fills missing data with a custom value

- **dropna()**
  - Removes rows or columns with missing values — useful when missingness is minimal and random

---

## 🎨 2. Encoding Categorical Variables
> ML models need numerical input. Categorical data must be converted first.

- **OneHotEncoder**
  - Converts categories into binary vectors (e.g. `Male`, `Female` becomes two separate columns)
  - Prevents the model from assuming any order or weight between categories

- **LabelEncoder**
  - Converts categories into integer labels (e.g. `Male` → 1, `Female` → 0)
  - Best for target variables, not for input features (can mislead models into assuming order)

---

## 📏 3. Feature Scaling
> Keeps features on a similar scale so models like SVMs and Logistic Regression perform better.

- **StandardScaler**
  - Transforms features to have a **mean of 0** and **standard deviation of 1**

- **MinMaxScaler**
  - Scales features to a fixed range, typically `[0, 1]`

---

## ✂️ 4. Feature Selection
> Removes unimportant or redundant features, helping reduce noise and improve performance.

- **VarianceThreshold**
  - Drops features with low variance — those that don’t change much across data points

---

## 🔽 5. Dimensionality Reduction
> Helps reduce computational cost and remove correlated redundancy in features.

- **PCA (Principal Component Analysis)**
  - Projects data into fewer dimensions while preserving most of the variance
  - Helpful for visualizations and avoiding overfitting

---

## 🔀 6. Data Splitting
> To evaluate a model properly, we must test it on data it hasn’t seen before.

- **train_test_split**
  - Divides data into training and testing sets
  - Usually 80% training, 20% testing (but customizable)

---

## 🧠 Tools & Libraries Used
- `pandas` for data handling
- `sklearn` for preprocessing, pipelines, and modeling
- `matplotlib` & `seaborn` for data visualization

---

## ✅ Summary

By mastering these preprocessing steps, I’ve built the foundation for:
- Cleaning raw datasets
- Preparing features for ML models
- Improving model accuracy and stability

These techniques are universally applicable across classification, regression, and unsupervised ML tasks.


