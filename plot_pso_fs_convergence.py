import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from mafese import Data, MhaSelector


def load_dataset(csv_path: str = "class.csv"):
    df = pd.read_csv(csv_path)
    if "Number of Bugs" not in df.columns:
        raise ValueError("Expected a 'Number of Bugs' target column in class.csv")

    feature_cols = [c for c in df.columns if c != "Number of Bugs"]
    X_encoded = []
    for col in feature_cols:
        series = df[col]
        if is_numeric_dtype(series):
            X_encoded.append(pd.to_numeric(series, errors="coerce").astype(float).to_numpy())
        else:
            codes, _ = pd.factorize(series)
            X_encoded.append(codes.astype(float))
    X = np.column_stack(X_encoded)
    y_raw = df["Number of Bugs"].values
    y = (y_raw > 0).astype(int)
    return X, y, feature_cols


def run_mafese_mha_feature_selection(X, y, epoch: int = 20, pop_size: int = 20, verbose: bool = True):
    data = Data(X, y)
    data.split_train_test(test_size=0.2, random_state=42)

    optimizer_paras = {"epoch": epoch, "pop_size": pop_size}
    selector = MhaSelector(
        problem="classification",
        obj_name=None,
        estimator="rf",
        estimator_paras={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        optimizer="OriginalWOA",
        optimizer_paras=optimizer_paras,
        mode="single",
        n_workers=None,
        termination=None,
        seed=42,
        verbose=verbose,
    )

    selector.fit(data.X_train, data.y_train)

    indexes = np.asarray(selector.selected_feature_indexes, dtype=int)
    if indexes.size == 0:
        raise RuntimeError("MAFESE did not select any features.")

    best_info = selector.get_best_information()
    k = int(best_info.get("n_columns", indexes.size))
    f1 = float(best_info.get("fit", 0.0))

    best_f1_by_k: dict[int, float] = {k: f1}
    return best_f1_by_k, indexes


def plot_mafese_feature_selection_convergence(output_path: str = "pso_feature_selection_convergence.png"):
    X, y, feature_names = load_dataset("class.csv")
    best_f1_by_k, selected_indexes = run_mafese_mha_feature_selection(
        X, y, epoch=20, pop_size=20, verbose=True
    )

    feature_counts = np.array(sorted(best_f1_by_k.keys()))
    f1_scores = np.array([best_f1_by_k[k] for k in feature_counts])

    best_idx = int(np.argmax(f1_scores))
    convergence_features = int(feature_counts[best_idx])
    convergence_f1 = float(f1_scores[best_idx])

    # Figure 1: convergence curve (kept from previous version)
    plt.figure(figsize=(8, 5))
    plt.plot(feature_counts, f1_scores, marker="o", linewidth=2, label="F1 score")

    plt.scatter(
        [convergence_features],
        [convergence_f1],
        color="red",
        zorder=5,
        label="Điểm hội tụ (convergence point)",
    )
    plt.axvline(
        x=convergence_features,
        color="red",
        linestyle="--",
        alpha=0.7,
    )

    plt.annotate(
        f"Điểm hội tụ\n{convergence_features} features\nF1 ≈ {convergence_f1:.3f}",
        xy=(convergence_features, convergence_f1),
        xytext=(convergence_features + 1, convergence_f1 - 0.01),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        ha="left",
        va="top",
    )

    plt.xlabel("Số lượng features được chọn (Number of features)")
    plt.ylabel("F1 score")
    plt.title("Hội tụ Feature Selection dùng MAFESE (MHA)\n(MAFESE Feature Selection Convergence)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    # Figure 2: top-15 feature importances for selected features
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    X_sel = X[:, selected_indexes]
    rf.fit(X_sel, y)

    importances = rf.feature_importances_
    feat_names_sel = np.array([feature_names[i] for i in selected_indexes])

    order = np.argsort(importances)[::-1]
    top_k = min(15, order.size)
    order = order[:top_k]

    top_importances = importances[order]
    top_names = feat_names_sel[order]

    plt.figure(figsize=(8, 5))
    y_pos = np.arange(top_k)
    # Reverse so the most important feature is at the top
    plt.barh(y_pos, top_importances[::-1])
    plt.yticks(y_pos, top_names[::-1])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig("top15_feature_importances.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_mafese_feature_selection_convergence()
