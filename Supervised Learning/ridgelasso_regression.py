import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and prepare your data
# Assuming your CSV is already loaded and cleaned
# df = pd.read_csv('your_file.csv')
# df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
# df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)

# Prepare features (X) and target (y)
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Scale the features for Ridge/Lasso
# Ridge and Lasso are sensitive to feature scales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 1. REGULAR LINEAR REGRESSION (for comparison)
# =============================================================================
print("1. REGULAR LINEAR REGRESSION")
print("="*50)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr.predict(X_test_scaled)

# Metrics
lr_train_r2 = lr.score(X_train_scaled, y_train)
lr_test_r2 = lr.score(X_test_scaled, y_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)

print(f"Training R²: {lr_train_r2:.4f}")
print(f"Testing R²: {lr_test_r2:.4f}")
print(f"MSE: {lr_mse:.0f}")
print(f"Coefficients: {lr.coef_}")
print()

# =============================================================================
# 2. RIDGE REGRESSION
# =============================================================================
print("2. RIDGE REGRESSION")
print("="*50)

# Ridge adds L2 penalty: minimizes (RSS + α * Σβ²)
# α (alpha) controls regularization strength
# Higher α = more regularization = smaller coefficients

# Try different alpha values
alphas = [0.1, 1.0, 10.0, 100.0]
ridge_results = {}

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    train_r2 = ridge.score(X_train_scaled, y_train)
    test_r2 = ridge.score(X_test_scaled, y_test)
    mse = mean_squared_error(y_test, y_pred_ridge)
    
    ridge_results[alpha] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mse': mse,
        'coefficients': ridge.coef_
    }
    
    print(f"Alpha = {alpha}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  MSE: {mse:.0f}")
    print(f"  Coefficients: {ridge.coef_}")
    print()

# =============================================================================
# 3. LASSO REGRESSION
# =============================================================================
print("3. LASSO REGRESSION")
print("="*50)

# Lasso adds L1 penalty: minimizes (RSS + α * Σ|β|)
# L1 penalty can make coefficients exactly zero (feature selection)
# Higher α = more regularization = more coefficients become zero

lasso_results = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_train_scaled, y_train)
    
    y_pred_lasso = lasso.predict(X_test_scaled)
    
    train_r2 = lasso.score(X_train_scaled, y_train)
    test_r2 = lasso.score(X_test_scaled, y_test)
    mse = mean_squared_error(y_test, y_pred_lasso)
    
    lasso_results[alpha] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mse': mse,
        'coefficients': lasso.coef_
    }
    
    print(f"Alpha = {alpha}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  MSE: {mse:.0f}")
    print(f"  Coefficients: {lasso.coef_}")
    
    # Show which features were selected (non-zero coefficients)
    selected_features = [f for f, coef in zip(X.columns, lasso.coef_) if coef != 0]
    print(f"  Selected features: {selected_features}")
    print()

# =============================================================================
# 4. COMPARISON AND VISUALIZATION
# =============================================================================
print("4. COMPARISON SUMMARY")
print("="*50)

# Find best Ridge and Lasso models
best_ridge_alpha = min(ridge_results.keys(), key=lambda a: ridge_results[a]['mse'])
best_lasso_alpha = min(lasso_results.keys(), key=lambda a: lasso_results[a]['mse'])

print(f"Best Ridge (α={best_ridge_alpha}): Test R² = {ridge_results[best_ridge_alpha]['test_r2']:.4f}")
print(f"Best Lasso (α={best_lasso_alpha}): Test R² = {lasso_results[best_lasso_alpha]['test_r2']:.4f}")
print(f"Linear Regression: Test R² = {lr_test_r2:.4f}")

# Plot coefficient comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Ridge coefficients for different alphas
feature_names = X.columns
ridge_coefs = np.array([ridge_results[alpha]['coefficients'] for alpha in alphas])

for i, feature in enumerate(feature_names):
    ax1.plot(alphas, ridge_coefs[:, i], 'o-', label=feature)
ax1.set_xlabel('Alpha (Regularization Strength)')
ax1.set_ylabel('Coefficient Value')
ax1.set_title('Ridge: Coefficient Shrinkage')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True)

# Lasso coefficients for different alphas
lasso_coefs = np.array([lasso_results[alpha]['coefficients'] for alpha in alphas])

for i, feature in enumerate(feature_names):
    ax2.plot(alphas, lasso_coefs[:, i], 'o-', label=feature)
ax2.set_xlabel('Alpha (Regularization Strength)')
ax2.set_ylabel('Coefficient Value')
ax2.set_title('Lasso: Coefficient Selection')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# 5. WHEN TO USE WHICH METHOD
# =============================================================================
print("\n5. DECISION GUIDE")
print("="*50)
print("Use RIDGE when:")
print("- You want to keep all features but reduce their impact")
print("- Features are correlated (multicollinearity)")
print("- You have more features than data points")
print("- You want stable, consistent predictions")
print()
print("Use LASSO when:")
print("- You want automatic feature selection")
print("- You suspect many features are irrelevant")
print("- You want a simpler, more interpretable model")
print("- You have many features and want to identify the most important ones")
print()
print("Use REGULAR LINEAR REGRESSION when:")
print("- You have few features (like your case with 3 features)")
print("- No signs of overfitting")
print("- Features are not highly correlated")
print("- You want the simplest model that works")