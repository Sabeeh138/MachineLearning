import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=== SVM COMPLETE EXAMPLE ===\n")

# 1. CREATE SAMPLE DATA
print("1. Creating sample data...")
# Generate a simple 2D dataset
X, y = datasets.make_classification(
    n_samples=200, 
    n_features=2, 
    n_redundant=0, 
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")

# 2. VISUALIZE RAW DATA
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Class 0', alpha=0.7)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Class 1', alpha=0.7)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. SPLIT DATA
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 4. SCALE DATA (VERY IMPORTANT FOR SVM!)
print("\n3. Scaling data (crucial for SVM)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Before scaling - Training data stats:")
print(f"Feature 1: mean={X_train[:, 0].mean():.2f}, std={X_train[:, 0].std():.2f}")
print(f"Feature 2: mean={X_train[:, 1].mean():.2f}, std={X_train[:, 1].std():.2f}")

print("After scaling - Training data stats:")
print(f"Feature 1: mean={X_train_scaled[:, 0].mean():.2f}, std={X_train_scaled[:, 0].std():.2f}")
print(f"Feature 2: mean={X_train_scaled[:, 1].mean():.2f}, std={X_train_scaled[:, 1].std():.2f}")

# 5. TRAIN DIFFERENT SVM MODELS
print("\n4. Training different SVM models...")

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# RBF SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Polynomial SVM
svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
svm_poly.fit(X_train_scaled, y_train)

print("Models trained successfully!")

# 6. MAKE PREDICTIONS
print("\n5. Making predictions...")
y_pred_linear = svm_linear.predict(X_test_scaled)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
y_pred_poly = svm_poly.predict(X_test_scaled)

# 7. EVALUATE MODELS
print("\n6. Model Performance:")
print("\nLinear SVM:")
print(f"Accuracy: {svm_linear.score(X_test_scaled, y_test):.3f}")
print(f"Support Vectors: {svm_linear.n_support_}")
print(f"Total Support Vectors: {svm_linear.support_vectors_.shape[0]}")

print("\nRBF SVM:")
print(f"Accuracy: {svm_rbf.score(X_test_scaled, y_test):.3f}")
print(f"Support Vectors: {svm_rbf.n_support_}")
print(f"Total Support Vectors: {svm_rbf.support_vectors_.shape[0]}")

print("\nPolynomial SVM:")
print(f"Accuracy: {svm_poly.score(X_test_scaled, y_test):.3f}")
print(f"Support Vectors: {svm_poly.n_support_}")
print(f"Total Support Vectors: {svm_poly.support_vectors_.shape[0]}")

# 8. VISUALIZE DECISION BOUNDARIES
def plot_decision_boundary(X, y, model, title, subplot_pos):
    plt.subplot(2, 3, subplot_pos)
    
    # Create a mesh of points
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # Plot the data points
    scatter = plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Class 0', alpha=0.7)
    scatter = plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Class 1', alpha=0.7)
    
    # Plot support vectors
    support_vectors = model.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                s=100, facecolors='none', edgecolors='black', linewidth=2, 
                label='Support Vectors')
    
    plt.title(title)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot decision boundaries for all models
plot_decision_boundary(X_train_scaled, y_train, svm_linear, 'Linear SVM', 2)
plot_decision_boundary(X_train_scaled, y_train, svm_rbf, 'RBF SVM', 3)
plot_decision_boundary(X_train_scaled, y_train, svm_poly, 'Polynomial SVM', 4)

# 9. CONFUSION MATRICES
plt.subplot(2, 3, 5)
cm = confusion_matrix(y_test, y_pred_rbf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix (RBF SVM)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 10. DETAILED CLASSIFICATION REPORT
plt.subplot(2, 3, 6)
plt.text(0.1, 0.8, "Classification Report (RBF SVM):", fontsize=12, fontweight='bold')
report = classification_report(y_test, y_pred_rbf, output_dict=True)
report_text = f"""
Precision Class 0: {report['0']['precision']:.3f}
Recall Class 0: {report['0']['recall']:.3f}
F1-Score Class 0: {report['0']['f1-score']:.3f}

Precision Class 1: {report['1']['precision']:.3f}
Recall Class 1: {report['1']['recall']:.3f}
F1-Score Class 1: {report['1']['f1-score']:.3f}

Overall Accuracy: {report['accuracy']:.3f}
"""
plt.text(0.1, 0.1, report_text, fontsize=10, verticalalignment='bottom')
plt.axis('off')

plt.tight_layout()
plt.show()

# 11. UNDERSTANDING SVM PARAMETERS
print("\n7. Understanding SVM Parameters:")
print("\nC Parameter (Regularization):")
print("- Higher C: Less regularization, tries to classify all points correctly")
print("- Lower C: More regularization, allows some misclassification for better generalization")

print("\nKernel Types:")
print("- Linear: Works best when data is linearly separable")
print("- RBF: Good general-purpose kernel, creates circular decision boundaries")
print("- Polynomial: Creates curved boundaries, good for complex patterns")

print("\nGamma (for RBF kernel):")
print("- Higher gamma: More complex decision boundary, risk of overfitting")
print("- Lower gamma: Simpler decision boundary, better generalization")

# 12. PARAMETER TUNING EXAMPLE
print("\n8. Quick Parameter Tuning Example:")
C_values = [0.1, 1, 10, 100]
accuracies = []

for C in C_values:
    svm_temp = SVC(kernel='rbf', C=C, random_state=42)
    svm_temp.fit(X_train_scaled, y_train)
    acc = svm_temp.score(X_test_scaled, y_test)
    accuracies.append(acc)
    print(f"C = {C}: Accuracy = {acc:.3f}")

best_C = C_values[np.argmax(accuracies)]
print(f"\nBest C value: {best_C} with accuracy: {max(accuracies):.3f}")

print("\n=== KEY TAKEAWAYS ===")
print("1. Always scale your data before using SVM!")
print("2. Support vectors are the only points that matter for the decision boundary")
print("3. Different kernels work better for different types of data")
print("4. Tune C parameter to balance between training accuracy and generalization")
print("5. RBF kernel is often a good starting point")
print("6. More support vectors might indicate overfitting or complex data")