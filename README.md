# Breast Cancer Classification using K-Nearest Neighbors

A machine learning model that classifies breast tumors as benign or malignant using the K-Nearest Neighbors algorithm on the Wisconsin Breast Cancer Dataset.

## 📊 Dataset
- **Source**: Scikit-learn built-in dataset
- **Features**: 30 numerical features
- **Samples**: 569 instances
- **Classes**: Binary classification (0 = Benign, 1 = Malignant)
- **Distribution**: 357 malignant, 212 benign

## 🛠️ Methodology
- **Data Split**: 70% training, 30% testing
- **Preprocessing**: Feature scaling using StandardScaler
- **Algorithm**: K-Nearest Neighbors Classifier
- **Evaluation Metrics**: Precision and Recall (focus on malignant class)

## 📈 Model Performance

| k-value | Precision | Recall |
|---------|-----------|--------|
| k=3     | 0.930     | 1.000  |
| k=5     | 0.939     | 1.000  |
| k=7     | 0.947     | 1.000  |
| k=9     | 0.947     | 1.000  |
| k=11    | 0.947     | 1.000  |

## ✨ Key Findings
- **Perfect Recall**: All k-values achieved 100% recall, correctly identifying all malignant cases
- **Optimal Performance**: k=7, 9, and 11 show the highest precision (94.7%)
- **Clinical Relevance**: High recall ensures no malignant cases are missed, crucial for medical diagnosis

## 🚀 Quick Start

```python
# Example usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
