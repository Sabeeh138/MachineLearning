import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Train a model (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)  # Works for tree-based models
shap_values = explainer.shap_values(X_test)

# 1. Summary Plot (Global Feature Importance)
print("\nGlobal Feature Importance (SHAP Values)")
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, plot_type="bar")

# 2. Force Plot (Single Prediction Explanation)
print("\nExplanation for 1st Test Sample:")
sample_idx = 0  # First test sample
shap.force_plot(
    explainer.expected_value[1],  # Base value (avg prediction)
    shap_values[1][sample_idx],   # SHAP values for this sample
    X_test[sample_idx],           # Feature values
    feature_names=feature_names
)

# 3. Beeswarm Plot (Detailed Distribution)
print("\nDetailed SHAP Values for All Features:")
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)