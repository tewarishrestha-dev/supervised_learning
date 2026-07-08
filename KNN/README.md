# K-Nearest Neighbors (KNN) From Scratch Using NumPy

A complete implementation of the K-Nearest Neighbors algorithm from scratch using NumPy.  
The project focuses on understanding the mathematical idea behind distance-based classification and compares the implementation with Scikit-Learn's KNN classifier.

The algorithm is implemented without using machine learning libraries for training or prediction.

---

# Project Overview

K-Nearest Neighbors is a simple yet powerful supervised learning algorithm that classifies a data point based on the labels of its closest neighbors.

Unlike parametric models, KNN does not learn a fixed equation during training. Instead, it stores the training data and performs distance calculations during prediction.

The implementation covers:

- KNN classification from scratch
- Different distance metrics
- Feature scaling
- Selection of optimal K value
- Weighted KNN
- Comparison with Scikit-Learn

---

# Dataset

The project uses the Breast Cancer Wisconsin Diagnostic Dataset provided by Scikit-Learn.

Dataset details:

- Samples: 569
- Features: 30
- Classes:
  - Malignant
  - Benign

Before applying KNN, features are standardized using Standard Scaling:

\[
z=\frac{x-\mu}{\sigma}
\]

where:

- \(x\) = original feature value
- \(\mu\) = mean of the feature
- \(\sigma\) = standard deviation

Feature scaling is important because KNN depends entirely on distance calculations.

---

# Mathematical Foundation

## Distance Between Points

KNN classifies samples based on similarity. Similarity is measured using distance.

For two points:

\[
x=(x_1,x_2,...,x_n)
\]

and

\[
y=(y_1,y_2,...,y_n)
\]

the distance is calculated using different metrics.

---

# Euclidean Distance

The most commonly used distance metric:

\[
d(x,y)=
\sqrt{
\sum_{i=1}^{n}(x_i-y_i)^2
}
\]

It represents the straight-line distance between two points.

---

# Manhattan Distance

Also known as L1 distance:

\[
d(x,y)=
\sum_{i=1}^{n}|x_i-y_i|
\]

It measures the total movement along each dimension.

---

# Minkowski Distance

A generalized distance metric:

\[
d(x,y)=
\left(
\sum_{i=1}^{n}|x_i-y_i|^p
\right)^{\frac1p}
\]

Different values of \(p\) produce different distances:

\[
p=1 \rightarrow Manhattan
\]

\[
p=2 \rightarrow Euclidean
\]

---

# Chebyshev Distance

It considers only the largest difference between features:

\[
d(x,y)=
\max_i |x_i-y_i|
\]

---

# KNN Algorithm

Given a new data point:

1. Calculate distance from the point to every training sample.

\[
d(x,x_i)
\]

2. Sort all distances.

3. Select the closest \(K\) samples.

4. Predict the class using majority voting.

For classification:

\[
\hat y =
\operatorname{mode}(y_1,y_2,...,y_k)
\]

where \(k\) represents the number of nearest neighbors.

---

# Choosing the Value of K

The value of K controls the behavior of the classifier.

Small K:

\[
K \rightarrow 1
\]

- Low bias
- High variance
- Sensitive to noise


Large K:

- Higher bias
- Smoother decision boundary
- May ignore important local patterns


The optimal value balances bias and variance.

\[
Error =
Bias^2 + Variance + Noise
\]

The project analyzes different K values and selects the value producing the highest validation accuracy.

---

# Weighted KNN

Standard KNN gives equal importance to every neighbor.

Weighted KNN assigns higher importance to closer points.

The weight is calculated as:

\[
w_i=
\frac{1}{d_i+\epsilon}
\]

where:

- \(d_i\) = distance of the neighbor
- \(\epsilon\) = small value to avoid division by zero

Closer neighbors contribute more strongly during voting.

---

# Complexity Analysis

For a dataset with:

- \(n\) training samples
- \(d\) features

Distance calculation requires:

\[
O(nd)
\]

operations for one query point.

Sorting the distances requires approximately:

\[
O(n\log n)
\]

Therefore, KNN has low training cost but expensive prediction because distances must be computed during inference.

---

# Experiments and Visualizations

The project includes:

- Accuracy comparison with Scikit-Learn KNN
- Accuracy vs K analysis
- Distance metric comparison
- Weighted KNN comparison
- Neighbor distance distribution

---

# Conclusion

This project demonstrates the mathematical working of KNN, from distance calculation to neighbor selection and classification.

Implementing KNN from scratch provides a deeper understanding of why feature scaling, distance metrics, and the choice of K value strongly affect model performance.
