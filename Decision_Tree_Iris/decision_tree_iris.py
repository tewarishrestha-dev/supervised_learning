from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Load Dataset

iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
print("First 5 Rows")
print(df.head())
print("\nShape of Dataset:", df.shape)
print("\nFeatures:", iris.feature_names)
print("Classes:", iris.target_names)

#Features and Target

X = iris.data
y = iris.target

#Splitting of Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

#Decision Tree

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

#Predictions

y_pred = model.predict(X_test)

#Accuracy

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("\nTrain Accuracy =", round(train_accuracy, 3))
print("Test Accuracy =", round(test_accuracy, 3))

#Classification 

print("\nClassification report\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

#Tree visualization

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree on Iris Dataset")
plt.tight_layout()
plt.savefig(
    "tree.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#Feature importance

importance = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.bar(
    iris.feature_names,
    importance
)
plt.xticks(rotation=45)
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#Overfitting 

depths = range(1, 20)
train_scores = []
test_scores = []

for d in depths:
    temp_model = DecisionTreeClassifier(
        max_depth=d,
        random_state=42
    )
    temp_model.fit(X_train, y_train)
    train_scores.append(
        temp_model.score(X_train, y_train)
    )
    test_scores.append(
        temp_model.score(X_test, y_test)
    )
plt.figure(figsize=(8, 5))
plt.plot(
    depths,
    train_scores,
    marker="o",
    label="Train Accuracy"
)
plt.plot(
    depths,
    test_scores,
    marker="o",
    label="Test Accuracy"
)
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "accuracy_vs_depth.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#Depthwise accuracy on training and test data

print("\nDepthwise accuracy")
for d, train_acc, test_acc in zip(
    depths,
    train_scores,
    test_scores
):
    print(
        f"Depth={d:2d} | "
        f"Train={train_acc:.3f} | "
        f"Test={test_acc:.3f}"
    )
