import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mealpy import FloatVar
from mealpy.swarm_based import WOA


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
    return X, y


def build_woa_objective(X, y, best_f1_by_k: dict):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # These dicts are mutated inside the closure for simple stateful logging
    eval_counter = {"n": 0}
    global_best = {"f1": -1.0, "k": None}

    def objective(solution):
        sol = np.asarray(solution)
        mask = sol > 0.5
        if not np.any(mask):
            mask[np.argmax(sol)] = True

        X_sel = X[:, mask]
        scores = cross_val_score(clf, X_sel, y, cv=skf, scoring="f1")
        f1 = float(scores.mean())

        k = X_sel.shape[1]
        prev = best_f1_by_k.get(k)
        if prev is None or f1 > prev:
            best_f1_by_k[k] = f1

        eval_counter["n"] += 1
        if f1 > global_best["f1"]:
            global_best["f1"] = f1
            global_best["k"] = k
            print(f"[WOA-FS] eval={eval_counter['n']:4d}  NEW global best F1={f1:.4f} with k={k}")
        elif eval_counter["n"] % 20 == 0:
            print(f"[WOA-FS] eval={eval_counter['n']:4d}  F1={f1:.4f} with k={k}")

        return 1.0 - f1

    return objective


def run_woa_feature_selection(X, y, epoch: int = 20, pop_size: int = 20, verbose: bool = True):
    n_features = X.shape[1]
    best_f1_by_k: dict[int, float] = {}
    obj_func = build_woa_objective(X, y, best_f1_by_k)

    bounds = FloatVar(lb=(0.0,) * n_features, ub=(1.0,) * n_features, name="w")
    problem = {
        "bounds": bounds,
        "obj_func": obj_func,
        "minmax": "min",
        "name": "WOA Feature Selection (F1)",
        "log_to": None,
    }

    print(f"[WOA-FS] Starting WOA: n_features={n_features}, epoch={epoch}, pop_size={pop_size}")
    model = WOA.OriginalWOA(epoch=epoch, pop_size=pop_size, verbose=verbose)
    model.solve(problem)

    print("[WOA-FS] Finished WOA. Best F1 by feature count (k):")
    for k in sorted(best_f1_by_k.keys()):
        print(f"  k={k:3d} -> F1={best_f1_by_k[k]:.4f}")

    return best_f1_by_k


def plot_woa_feature_selection_convergence(output_path: str = "pso_feature_selection_convergence.png"):
    X, y = load_dataset("class.csv")
    best_f1_by_k = run_woa_feature_selection(X, y, epoch=20, pop_size=20, verbose=True)

    feature_counts = np.array(sorted(best_f1_by_k.keys()))
    f1_scores = np.array([best_f1_by_k[k] for k in feature_counts])

    best_idx = int(np.argmax(f1_scores))
    convergence_features = int(feature_counts[best_idx])
    convergence_f1 = float(f1_scores[best_idx])

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
    plt.title("Hội tụ Feature Selection dùng PSO + Mealpy\n(PSO Feature Selection Convergence)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_woa_feature_selection_convergence()
