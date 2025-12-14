from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_used_cars_results(
    *,
    results_df: pd.DataFrame,
    df_results: pd.DataFrame,
    cv_long_df: pd.DataFrame,
    dataset_name: str = "used cars",
    y_test: np.ndarray,
    preds_test: dict[str, np.ndarray],
    models_store: dict[str, object] | None = None,
    X_test: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    top_k_models: int = 3,
) -> None:
    sns.set_theme(style="whitegrid")

    ds = (dataset_name or "").strip()
    prefix = f"{ds}: " if ds else ""

    holdout_long = results_df.melt(
        id_vars=["model_name"],
        value_vars=["rmse", "rsquared"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(12, 4))
    ax = sns.barplot(
        data=holdout_long[holdout_long["metric"] == "rmse"],
        x="model_name",
        y="value",
    )
    plt.title(f"{prefix}Hold-out: RMSE by model")
    # value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    ax = sns.barplot(
        data=holdout_long[holdout_long["metric"] == "rsquared"],
        x="model_name",
        y="value",
    )
    plt.title(f"{prefix}Hold-out: R² by model")
    plt.ylim(0, 1)
    # value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    cv_long_melt = cv_long_df.melt(
        id_vars=["model_name", "fold"],
        value_vars=["rmse", "r2"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(12, 4))
    sns.boxplot(
        data=cv_long_melt[cv_long_melt["metric"] == "rmse"],
        x="model_name",
        y="value",
    )
    plt.title(f"{prefix}10-fold CV: RMSE distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    sns.boxplot(
        data=cv_long_melt[cv_long_melt["metric"] == "r2"],
        x="model_name",
        y="value",
    )
    plt.title(f"{prefix}10-fold CV: R² distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    top_models = (
        results_df.sort_values("rmse", ascending=True)
        .head(int(top_k_models))["model_name"]
        .tolist()
    )

    for name in top_models:
        y_pred = preds_test[name]
        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].scatter(y_test, y_pred, s=12, alpha=0.5)
        lo = float(min(np.min(y_test), np.min(y_pred)))
        hi = float(max(np.max(y_test), np.max(y_pred)))
        axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1)
        axes[0].set_title(f"{prefix}{name}\nActual vs Pred")
        axes[0].set_xlabel("y")
        axes[0].set_ylabel("ŷ")

        # residuals vs predicted
        axes[1].scatter(y_pred, residuals, s=12, alpha=0.5)
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_title("Residuals vs Pred")
        axes[1].set_xlabel("ŷ")
        axes[1].set_ylabel("y-ŷ")

        # residual distribution
        sns.histplot(residuals, kde=True, ax=axes[2])
        axes[2].set_title("Residual distribution")
        axes[2].set_xlabel("y-ŷ")

        plt.tight_layout()
        plt.show()

    if models_store is not None:
        pruned_name = "MyDecisionTreeRegressor post-pruned"
        if pruned_name in models_store:
            pruned_model = models_store[pruned_name]
            alphas = getattr(pruned_model, "alpha_candidates", None)
            mse_vals = getattr(pruned_model, "alpha_cv_errors", None)
            best_alpha = getattr(pruned_model, "best_alpha", None)

            if alphas is not None and mse_vals is not None and len(alphas) == len(mse_vals):
                rmse_vals = np.sqrt(np.asarray(mse_vals))
                plt.figure(figsize=(8, 4))
                plt.plot(alphas, rmse_vals, marker="o", linewidth=1)
                if best_alpha is not None:
                    plt.axvline(best_alpha, color="red", linestyle="--", label=f"best_alpha={best_alpha:.4g}")
                    plt.legend()
                plt.title(f"{prefix}MyDecisionTreeRegressor: post-pruning path")
                plt.xlabel("alpha")
                plt.ylabel("CV RMSE")
                plt.tight_layout()
                plt.show()

    if models_store is not None and feature_names is not None:
        ridge_name = "Linear Regression"
        model = models_store.get(ridge_name)
        if model is not None and hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).ravel()
            s = (
                pd.Series(coef, index=feature_names)
                .sort_values(key=lambda x: np.abs(x), ascending=False)
                .head(15)
            )
            s.index.name = "Features"
            plt.figure(figsize=(8, 5))
            sns.barplot(x=s.values, y=s.index, orient="h")
            plt.title("Ridge: top |coefficients| (scaled features)")
            plt.tight_layout()
            plt.show()

        for name in ["Decision Tree", "Random Forest (10 Trees)"]:
            model = models_store.get(name)
            if model is not None and hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_).ravel()
                s = (
                    pd.Series(imp, index=feature_names)
                    .sort_values(ascending=False)
                    .head(15)
                )
                s.index.name = "Features"
                plt.figure(figsize=(8, 5))
                sns.barplot(x=s.values, y=s.index, orient="h")
                plt.title(f"{name}: top feature importances")
                plt.tight_layout()
                plt.show()

    if models_store is not None and X_test is not None:
        sk_name = "Random Forest (10 Trees)"
        rf = models_store.get(sk_name)
        if rf is not None and hasattr(rf, "estimators_"):
            per_tree = np.column_stack([t.predict(X_test) for t in rf.estimators_])
            std = per_tree.std(axis=1)
            mean = per_tree.mean(axis=1)

            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=mean, y=std, s=18, alpha=0.6)
            plt.title(f"{prefix}sklearn RF: prediction mean vs per-tree std")
            plt.xlabel("mean prediction")
            plt.ylabel("std across trees")
            plt.tight_layout()
            plt.show()

        # best custom RF
        custom_names = [n for n in models_store.keys() if "MyRandomForestRegressor" in n]
        if custom_names:
            best_custom_name = (
                results_df[results_df["model_name"].isin(custom_names)]
                .sort_values("rmse", ascending=True)
                .iloc[0]["model_name"]
            )
            rf = models_store.get(best_custom_name)
            if rf is not None and hasattr(rf, "trees") and hasattr(rf, "feature_indices_per_tree"):
                X = np.asarray(X_test)
                all_preds = np.zeros((X.shape[0], len(rf.trees)), dtype=float)
                for i, (tree, feat_idx) in enumerate(zip(rf.trees, rf.feature_indices_per_tree)):
                    all_preds[:, i] = tree.predict(X[:, feat_idx])

                std = all_preds.std(axis=1)
                mean = all_preds.mean(axis=1)

                plt.figure(figsize=(6, 4))
                sns.scatterplot(x=mean, y=std, s=18, alpha=0.6)
                plt.title(f"{prefix}Custom RF: prediction mean vs per-tree std\n({best_custom_name})")
                plt.xlabel("mean prediction")
                plt.ylabel("std across trees")
                plt.tight_layout()
                plt.show()

