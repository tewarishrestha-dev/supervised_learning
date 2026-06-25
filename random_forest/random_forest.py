from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
mean_squared_error,
mean_absolute_error,
r2_score
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load Dataset

housing = fetch_california_housing()

df = pd.DataFrame(
housing.data,
columns=housing.feature_names
)

df["Price"] = housing.target

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

# Features and Target

X = housing.data
y = housing.target

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.3,
random_state=7
)

# Random Forest Model

rf = RandomForestRegressor(
n_estimators=100,
max_depth=10,
random_state=7,
n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Evaluation Metrics

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 30)
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Feature Importance

importance = rf.feature_importances_

importance_df = pd.DataFrame({
"Feature": housing.feature_names,
"Importance": importance
})

importance_df = importance_df.sort_values(
by="Importance",
ascending=False
)

print("\nFeature Importance")
print("-" * 30)
print(importance_df)

plt.figure(figsize=(8,5))

plt.bar(
importance_df["Feature"],
importance_df["Importance"]
)

plt.xticks(rotation=45)

plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
"feature_importance.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# Actual vs Predicted

plt.figure(figsize=(7,7))

plt.scatter(
y_test,
y_pred,
alpha=0.3
)

plt.plot(
[y_test.min(), y_test.max()],
[y_test.min(), y_test.max()],
color="red",
label="Perfect Prediction"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.legend()

plt.tight_layout()

plt.savefig(
"actual_vs_predicted.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# Residual Plot

residual = y_test - y_pred

plt.figure(figsize=(8,5))

plt.scatter(
y_pred,
residual,
alpha=0.3
)

plt.axhline(
y=0,
color="red"
)

plt.xlabel("Predicted")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
"residual_plot.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# Residual Histogram

plt.figure(figsize=(8,5))

plt.hist(
residual,
bins=50
)

plt.xlabel("Residual")
plt.ylabel("Frequency")

plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig(
"residual_histogram.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# Decision Tree vs Random Forest

tree = DecisionTreeRegressor(
max_depth=10,
random_state=7
)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

tree_r2 = r2_score(
y_test,
tree_pred
)

forest_r2 = r2_score(
y_test,
y_pred
)

print("\nModel Comparison")
print("-" * 30)
print(f"Decision Tree R² : {tree_r2:.4f}")
print(f"Random Forest R² : {forest_r2:.4f}")

models = ["Decision Tree", "Random Forest"]
comparison_scores = [tree_r2, forest_r2]

plt.figure(figsize=(6,5))

plt.bar(
models,
comparison_scores
)

plt.ylabel("R² Score")

plt.title("Decision Tree vs Random Forest")

plt.tight_layout()

plt.savefig(
"model_comparison.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# Number of Trees Analysis

estimators = [1, 5, 10, 20, 50, 100, 200]

estimator_scores = []

for n in estimators:

    temp_rf = RandomForestRegressor(
    n_estimators=n,
    random_state=7,
    n_jobs=-1
)

    temp_rf.fit(
    X_train,
    y_train
)

    estimator_scores.append(
    temp_rf.score(
        X_test,
        y_test
    )
)


plt.figure(figsize=(8,5))

plt.plot(
estimators,
estimator_scores,
marker="o"
)

plt.xlabel("Number of Trees")
plt.ylabel("R² Score")

plt.title("R² vs Number of Trees")

plt.grid()

plt.tight_layout()

plt.savefig(
"r2_vs_estimators.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# Overfitting Analysis

depths = range(1, 21)

train_scores = []
test_scores = []

for d in depths:

    temp_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=d,
    random_state=7,
    n_jobs=-1
)

    temp_rf.fit(
    X_train,
    y_train
)

    train_scores.append(
    temp_rf.score(
        X_train,
        y_train
    )
)

    test_scores.append(
    temp_rf.score(
        X_test,
        y_test
    )
)


plt.figure(figsize=(8,5))

plt.plot(
depths,
train_scores,
marker="o",
label="Train R²"
)

plt.plot(
depths,
test_scores,
marker="o",
label="Test R²"
)

plt.xlabel("Max Depth")
plt.ylabel("R² Score")

plt.title("Overfitting Analysis")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
"r2_vs_depth.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# Depth-wise Scores

print("\nDepth-wise R²")
print("-" * 40)

for d, train_r2, test_r2 in zip(
depths,
train_scores,
test_scores
):
    print(
        f"Depth={d:2d} | "
        f"Train={train_r2:.4f} | "
        f"Test={test_r2:.4f}"
    )
