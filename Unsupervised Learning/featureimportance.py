# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load the dataset
data = load_breast_cancer()
X = data.data  # Features (e.g., tumor radius, texture, etc.)
y = data.target  # Target (0 = malignant, 1 = benign)
feature_names = data.feature_names  # Names of the features

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Get feature importances (how much each feature matters)
feature_importance = model.feature_importances_

# 5. Sort features by importance (most important first)
sorted_idx = np.argsort(feature_importance)[::-1]  # [::-1] reverses order (descending)

# 6. Print the top 10 most important features
print("Top 10 Most Important Features:")
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: {feature_importance[i]:.4f}")

# 7. Plot feature importance (bar chart)
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Breast Cancer Dataset)")
plt.tight_layout()
plt.show()