import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from fractal_growth import FractalGrowthClassifier2D


def plot_model(model, X_train, y_train, title, output_path):
    x_min, x_max = X_train[:, 0].min() - 0.4, X_train[:, 0].max() + 0.4
    y_min, y_max = X_train[:, 1].min() - 0.4, X_train[:, 1].max() + 0.4

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, alpha=0.25)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=18, alpha=0.75)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    X, y = make_moons(n_samples=1000, noise=0.22, random_state=7)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=7,
    )

    fractal = FractalGrowthClassifier2D(
        max_depth=7,
        min_points=12,
        impurity_threshold=0.03,
    )

    fractal.fit(X_train, y_train)
    fractal_acc = accuracy_score(y_test, fractal.predict(X_test))

    neural = MLPClassifier(
        hidden_layer_sizes=(24, 24),
        activation="relu",
        max_iter=500,
        random_state=7,
    )

    neural.fit(X_train, y_train)
    neural_acc = accuracy_score(y_test, neural.predict(X_test))

    print("Fractal Growth accuracy:", round(fractal_acc, 4))
    print("Neural Network accuracy:", round(neural_acc, 4))
    print("Fractal growth steps:", len(fractal.history))

    plot_model(
        fractal,
        X_train,
        y_train,
        f"Fractal Growth Classifier | Accuracy: {fractal_acc:.3f}",
        output_dir / "fractal_decision_boundary.png",
    )

    plot_model(
        neural,
        X_train,
        y_train,
        f"Neural Network Classifier | Accuracy: {neural_acc:.3f}",
        output_dir / "neural_decision_boundary.png",
    )

    print("Saved visuals to outputs/")


if __name__ == "__main__":
    main()