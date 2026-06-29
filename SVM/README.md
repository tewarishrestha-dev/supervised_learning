# Support Vector Machine from Scratch using Stochastic Gradient Descent

A complete implementation of a **Linear Support Vector Machine (SVM)** from scratch using **NumPy** and **Stochastic Gradient Descent (SGD)** without relying on machine learning libraries for optimization. The objective of this project was to understand the mathematical foundations of Support Vector Machines by implementing the complete training pipeline manually, including hinge loss, L2 regularization, gradient computation, and parameter updates.

The implementation is evaluated on the Breast Cancer Wisconsin Diagnostic Dataset and compared against Scikit-Learn's Linear SVM.

---

## Features

* Linear SVM implemented completely from scratch
* Stochastic Gradient Descent optimizer
* Hinge Loss implementation
* L2 Regularization
* Manual gradient computation
* Manual weight and bias updates
* Comparison with Scikit-Learn's implementation
* Training Loss visualization
* Margin Distribution visualization
* Feature Importance visualization
* Confusion Matrix
* Accuracy, Training Time and Prediction Time comparison

---

# Dataset

Dataset : Breast Cancer Wisconsin Diagnostic Dataset

Samples : 569

Features : 30

Classes :

* Benign
* Malignant

The original labels

```
{0,1}
```

are converted to

```
{-1,+1}
```

since the optimization problem for SVM is naturally formulated using binary labels of ±1.

---

# Mathematical Foundation

Given a training dataset

```
D = {(x₁,y₁),(x₂,y₂),...,(xₙ,yₙ)}
```

where

```
x ∈ ℝᵈ
```

and

```
y ∈ {-1,+1}
```

the objective is to learn a hyperplane that separates both classes while maximizing the distance between them.

---

## Hyperplane

The decision boundary of a Linear SVM is

```
wᵀx + b = 0
```

where

```
w = Weight Vector

x = Feature Vector

b = Bias
```

Prediction is performed using

```
f(x)=wᵀx+b
```

```
f(x) ≥ 0  →  +1

f(x) < 0  →  -1
```

---

## Functional Margin

For a training sample,

```
γ̂ = y(wᵀx+b)
```

Interpretation

* Positive → Correct classification
* Negative → Misclassification
* Larger value → Higher confidence

The functional margin depends on the scale of **w** and **b**, therefore it is not an absolute measure of distance.

---

## Geometric Margin

The geometric margin represents the perpendicular distance between a sample and the decision boundary.

```
γ = y(wᵀx+b) / ||w||
```

Unlike the functional margin, the geometric margin remains unchanged if the hyperplane parameters are multiplied by a constant.

Support Vector Machines maximize this quantity.

---

## Distance from a Point to a Hyperplane

For a line

```
Ax + By + C = 0
```

the perpendicular distance of a point

```
(x₀,y₀)
```

is

```
|Ax₀ + By₀ + C|
-----------------------
√(A²+B²)
```

Extending this idea to higher dimensions gives

```
|wᵀx+b|
----------
||w||
```

which forms the basis of the geometric margin.

---

# Hard Margin SVM

If the data is perfectly linearly separable,

```
y(wᵀx+b) ≥ 1
```

for every training sample.

The optimization problem becomes

```
min  ½ ||w||²
```

subject to

```
y(wᵀx+b) ≥ 1
```

Minimizing

```
||w||
```

maximizes

```
1 / ||w||
```

which is equivalent to maximizing the geometric margin.

Hard Margin SVM assumes perfect separation and is highly sensitive to noisy observations.

---

# Soft Margin SVM

Real-world datasets are rarely perfectly separable.

Slack variables allow controlled violations of the margin.

The optimization objective becomes

```
min

½||w||² + C Σξᵢ
```

subject to

```
y(wᵀx+b) ≥ 1−ξᵢ
```

where

```
ξᵢ ≥ 0
```

The regularization parameter **C** controls the trade-off between maximizing the margin and minimizing classification errors.

Large values of **C** prioritize correct classification.

Smaller values of **C** allow a wider margin and better generalization.

# Optimization Objective

The constrained optimization problem of Soft Margin SVM can be converted into an unconstrained optimization problem by introducing the **Hinge Loss**.

The objective optimized during training is

```text
                λ
J(w,b) =  ───────── ||w||² + Σ max(0, 1 − y(wᵀx+b))
                 2
```

where

* λ : Regularization Parameter
* ||w||² : L2 Regularization
* max(0, ·) : Hinge Loss

The objective consists of two competing terms.

The first minimizes the magnitude of the weight vector, resulting in a larger geometric margin.

The second penalizes incorrectly classified samples and samples lying inside the margin.

The optimization therefore balances **model complexity** and **classification accuracy**.

---

# Hinge Loss

Unlike Logistic Regression, which uses Log Loss, Support Vector Machines optimize the **Hinge Loss**.

For a single training sample,

```text
L = max(0, 1 − y(wᵀx+b))
```

Depending on the value of

```text
y(wᵀx+b)
```

three situations arise.

### Case 1

```text
y(wᵀx+b) ≥ 1
```

* Correctly classified
* Outside the margin
* Hinge Loss = 0

Only the regularization term contributes to the gradient.

---

### Case 2

```text
0 < y(wᵀx+b) < 1
```

* Correctly classified
* Inside the margin
* Hinge Loss > 0

The optimizer attempts to push the sample farther away from the decision boundary.

---

### Case 3

```text
y(wᵀx+b) < 0
```

* Misclassified
* Largest contribution to the loss

These samples receive the largest parameter updates during optimization.

---

# Gradient Computation

Since the Hinge Loss is piecewise-defined, its gradient is also piecewise.

### If

```text
y(wᵀx+b) ≥ 1
```

```text
∂J/∂w = λw

∂J/∂b = 0
```

Only the regularization term contributes.

---

### If

```text
y(wᵀx+b) < 1
```

```text
∂J/∂w = λw − yx

∂J/∂b = −y
```

Both regularization and hinge loss contribute to the gradient.

---

# Stochastic Gradient Descent

Instead of computing gradients over the complete dataset, the model updates the parameters after processing every training sample.

For samples satisfying the margin,

```text
w ← w − η(λw)
```

For samples violating the margin,

```text
w ← w − η(λw − yx)

b ← b + ηy
```

where

```text
η
```

is the learning rate.

Repeating these updates over multiple epochs gradually minimizes the optimization objective.

---

# Results

The implementation was compared against Scikit-Learn's Linear Support Vector Machine using the same training and testing split.

Evaluation metrics include

* Classification Accuracy
* Confusion Matrix
* Training Time
* Prediction Time

The custom implementation achieves performance comparable to Scikit-Learn while exposing every mathematical step involved in the optimization process.

---

# Visualizations

The following visualizations were generated during training and evaluation.

## Training Loss

Illustrates the decrease in the optimization objective throughout training.

```
images/training_loss.png
```

---

## Margin Distribution

Displays the distribution of

```text
y(wᵀx+b)
```

for every training sample.

Interpretation

```text
Margin < 0
```

Misclassified samples

```text
0 ≤ Margin < 1
```

Correctly classified but inside the margin

```text
Margin ≥ 1
```

Correctly classified outside the margin

```
images/margin_distribution.png
```

---

## Feature Importance

For a Linear Support Vector Machine, the magnitude of each learned weight

```text
|w|
```

indicates the relative importance of the corresponding feature.

```
images/feature_importance.png
```

---

## Confusion Matrix

```
images/confusion_matrix_ours.png

images/confusion_matrix_sklearn.png
```

---

## Accuracy Comparison

```
images/accuracy_comparison.png
```

---

## Training Time Comparison

```
images/training_time.png
```

## Prediction Time Comparison
images/prediction_time.png

Project Objective

The primary objective of this project was not to achieve the highest possible accuracy, but to understand how Support Vector Machines learn from data by implementing every component of the optimization process manually.

Rather than relying on machine learning libraries for training, the complete learning algorithm—including hinge loss, regularization, gradient computation, and parameter updates—was implemented from first principles to gain a deeper understanding of the mathematics behind Support Vector Machines.

⭐ If you found this repository useful, consider giving it a star.
