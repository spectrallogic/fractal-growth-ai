import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from fractal_growth import FractalGrowthClassifierND


def get_datasets():
    return {
        "breast_cancer": load_breast_cancer(),
        "wine": load_wine(),
        "digits": load_digits(),
    }


def get_fractal_settings(dataset_name):
    settings = {
        "breast_cancer": {
            "max_depth": 14,
            "min_points": 5,
            "impurity_threshold": 0.002,
            "n_candidate_splits": 18,
            "max_features": "auto",
            "min_gain": 1e-5,
            "random_state": 42,
        },
        "wine": {
            "max_depth": 12,
            "min_points": 3,
            "impurity_threshold": 0.001,
            "n_candidate_splits": 18,
            "max_features": "all",
            "min_gain": 1e-5,
            "random_state": 42,
        },
        "digits": {
            "max_depth": 20,
            "min_points": 3,
            "impurity_threshold": 0.001,
            "n_candidate_splits": 14,
            "max_features": "auto",
            "min_gain": 1e-5,
            "random_state": 42,
        },
    }

    return settings[dataset_name]


def get_models(dataset_name):
    return {
        "Fractal Growth ND v4": FractalGrowthClassifierND(
            **get_fractal_settings(dataset_name)
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
        ),
        "MLP Neural Net": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=700,
                random_state=42,
            ),
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(),
        ),
    }


def save_accuracy_chart(df, dataset_name, output_dir):
    sub = df[df["dataset"] == dataset_name].sort_values(
        "accuracy",
        ascending=False,
    )

    plt.figure(figsize=(9, 4.8))
    plt.bar(sub["model"], sub["accuracy"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy comparison on {dataset_name}")
    plt.tight_layout()

    output_path = output_dir / f"{dataset_name}_accuracy_comparison.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def save_train_time_chart(df, dataset_name, output_dir):
    sub = df[df["dataset"] == dataset_name].sort_values(
        "train_time_sec",
        ascending=True,
    )

    plt.figure(figsize=(9, 4.8))
    plt.bar(sub["model"], sub["train_time_sec"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Train time, seconds")
    plt.title(f"Training time comparison on {dataset_name}")
    plt.tight_layout()

    output_path = output_dir / f"{dataset_name}_train_time_comparison.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def add_speed_ratios(results):
    rows = []

    for dataset_name in results["dataset"].unique():
        sub = results[results["dataset"] == dataset_name]
        fractal_time = sub[sub["model"] == "Fractal Growth ND v4"][
            "train_time_sec"
        ].iloc[0]

        for _, row in sub.iterrows():
            row = row.copy()

            if row["model"] == "Fractal Growth ND v4":
                row["speed_vs_fractal"] = 1.0
            else:
                row["speed_vs_fractal"] = row["train_time_sec"] / fractal_time

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    rows = []

    for dataset_name, ds in get_datasets().items():
        X = ds.data
        y = ds.target

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )

        models = get_models(dataset_name)

        for model_name, model in models.items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            row = {
                "dataset": dataset_name,
                "features": X.shape[1],
                "classes": len(set(y)),
                "model": model_name,
                "accuracy": accuracy,
                "train_time_sec": train_time,
            }

            if model_name == "Fractal Growth ND v4":
                row["growth_steps"] = len(model.history)
                row["final_regions"] = model.get_num_regions()
            else:
                row["growth_steps"] = None
                row["final_regions"] = None

            rows.append(row)

    results = pd.DataFrame(rows)
    results = add_speed_ratios(results)

    results = results.sort_values(
        ["dataset", "accuracy"],
        ascending=[True, False],
    )

    csv_path = output_dir / "standard_benchmark_results.csv"
    results.to_csv(csv_path, index=False)

    for dataset_name in get_datasets().keys():
        save_accuracy_chart(results, dataset_name, output_dir)
        save_train_time_chart(results, dataset_name, output_dir)

    print("\nBenchmark results:")
    print(results.to_string(index=False))

    print("\nSpeed ratio meaning:")
    print("- speed_vs_fractal > 1.0 means that model is slower than Fractal Growth")
    print("- speed_vs_fractal < 1.0 means that model is faster than Fractal Growth")

    print("\nSaved:")
    print(f"- {csv_path}")
    print("- outputs/breast_cancer_accuracy_comparison.png")
    print("- outputs/wine_accuracy_comparison.png")
    print("- outputs/digits_accuracy_comparison.png")
    print("- outputs/breast_cancer_train_time_comparison.png")
    print("- outputs/wine_train_time_comparison.png")
    print("- outputs/digits_train_time_comparison.png")


if __name__ == "__main__":
    main()