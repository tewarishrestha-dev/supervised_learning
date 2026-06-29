import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


os.makedirs("images", exist_ok=True)


def plot_confusion_matrix(cm, title, filename):

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")

    plt.title(title)

    plt.savefig(f"images/{filename}")

    plt.close()


def plot_accuracy(our_acc, sklearn_acc):

    plt.figure(figsize=(6,5))

    models = ["Our SVM", "Sklearn"]

    accuracy = [our_acc*100, sklearn_acc*100]

    plt.bar(models, accuracy)

    plt.ylabel("Accuracy (%)")

    plt.title("Accuracy Comparison")

    plt.ylim(0,100)

    for i,v in enumerate(accuracy):
        plt.text(i,v+1,f"{v:.2f}%")

    plt.savefig("images/accuracy_comparison.png")

    plt.close()


def plot_training_time(our_time, sklearn_time):

    plt.figure(figsize=(6,5))

    models=["Our SVM","Sklearn"]

    times=[our_time,sklearn_time]

    plt.bar(models,times)

    plt.ylabel("Seconds")

    plt.title("Training Time Comparison")

    for i,v in enumerate(times):
        plt.text(i,v,f"{v:.5f}")

    plt.savefig("images/training_time.png")

    plt.close()


def plot_prediction_time(our_time, sklearn_time):

    plt.figure(figsize=(6,5))

    models=["Our SVM","Sklearn"]

    times=[our_time,sklearn_time]

    plt.bar(models,times)

    plt.ylabel("Seconds")

    plt.title("Prediction Time Comparison")

    for i,v in enumerate(times):
        plt.text(i,v,f"{v:.6f}")

    plt.savefig("images/prediction_time.png")

    plt.close()

def plot_training_loss(loss_history):

    plt.figure(figsize=(8,5))

    plt.plot(
        loss_history,
        color="royalblue",
        linewidth=2.5,
        label="Training Loss"
    )

    plt.scatter(
        0,
        loss_history[0],
        color="red",
        s=80,
        label="Start"
    )

    plt.scatter(
        len(loss_history)-1,
        loss_history[-1],
        color="green",
        s=80,
        label="End"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("images/training_loss.png")

    plt.close()

def plot_feature_importance(weights, feature_names):

    importance = np.abs(weights)

    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10,8))

    plt.barh(
        np.array(feature_names)[indices],
        importance[indices],
        color="royalblue"
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Absolute Weight")

    plt.ylabel("Feature")

    plt.title("Feature Importance (Linear SVM)")

    plt.tight_layout()

    plt.savefig("images/feature_importance.png")

    plt.close()    

def plot_margin_distribution(model, X, y):

    margins = y * (X @ model.w + model.b)

    plt.figure(figsize=(9,6))

    plt.hist(
        margins,
        bins=30,
        color="royalblue",
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(
        x=0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Decision Boundary"
    )

    plt.axvline(
        x=1,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Margin"
    )

    plt.xlabel("Margin  y(wᵀx + b)")
    plt.ylabel("Number of Samples")

    plt.title("Margin Distribution")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig("images/margin_distribution.png")

    plt.close()