"""
run_mdfitml_lasso.py
=====================
Lasso regression with nested leave-one-out cross-validation.

Structure
---------
  0. Output writing class
  1. Data loading and preparing
  2. Outer LOOCV loop  (progress printed to screen, fold i of N)
       └─ Inner LOOCV via LassoCV  →  selects best alpha per fold
       └─ Final Lasso refit with best alpha  →  prediction + coefficient harvest
  3. Aggregate outer-loop performance  (RMSE, MAE, R², Pearson r, Kendall τ)
       └─ 95 % bootstrap CIs for all five metrics  (10 000 resamples)
  4. Per-sample summary table
  5. Top-N features by mean |coefficient| across all outer folds
  6. OLS follow-up trained & evaluated (nested LOOCV) on the top-N features
  7. Side-by-side comparison of Lasso vs OLS
  8. Bar chart of OLS coefficients (green = positive, red = negative)

Output files
------------
  results_mdfitml.txt   – all performance tables (nothing printed to screen)
  plot_mdfitml.png      – bar chart of OLS top-feature coefficients

"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau, bootstrap
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



# ─────────────────────────────────────────────────────────────────────────────
# 0.  Writer – thin wrapper that sends output to a file instead of the screen
# ─────────────────────────────────────────────────────────────────────────────

class Writer:
    """
    Context manager that redirects all write() calls to a plain-text file.

    Usage:
        with Writer("results.txt") as out:
            out.write("hello")
            out.section("Performance")
            out.write(some_dataframe.to_string())

    All performance reporting functions accept an `out` parameter of this type.
    Progress prints (fold counters) still go to sys.stdout directly.
    """

    SEP_DOUBLE = "═" * 72
    SEP_SINGLE = "─" * 60

    def __init__(self, path: str):
        self.path = path
        self._fh  = None

    def __enter__(self):
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, *_):
        if self._fh:
            self._fh.close()

    def write(self, text: str = ""):
        """Write a line (newline appended automatically if missing)."""
        if not text.endswith("\n"):
            text += "\n"
        self._fh.write(text)

    def section(self, title: str, wide: bool = True):
        sep = self.SEP_DOUBLE if wide else self.SEP_SINGLE
        self.write(sep)
        self.write(f"  {title}")
        self.write(sep)

    def rule(self, wide: bool = False):
        self.write(self.SEP_DOUBLE if wide else self.SEP_SINGLE)

    def blank(self):
        self.write("")



# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_feat, data_obs) :

    # Load data
    df_feat = pd.read_csv(data_feat, sep=',', header=0, index_col=0)
    df_obs  = pd.read_csv(data_obs,  sep=',', header=0, index_col=0)


    # Merge data to clearly define feature matrix (X) and observed (response) variable (y)
    # for all data instances. First, redefine mol index column names for correct merging.
    # Upon merging, the same observed value may be added to multiple instances, which stem
    # from MD run repeats. Then remove any instance, for which data are missing.

    df_feat.index.name  = 'mol_name'
    df_obs.index.name   = 'mol_name'

    df_data = pd.merge(df_feat, df_obs, on='mol_name', how='left')

    df_data = df_data.dropna()

    #df_data = df_data.head(50) # for quicker debugging

    X = np.array(df_data.drop('potency', axis=1))
    y = np.array(df_data['potency'])
    sample_ids = np.array(df_data.index)
    feature_names = list(df_data.columns[:-1])
    # The appended column of observed data is taken off the list of feature names

    return (X, y, sample_ids, feature_names)



# ─────────────────────────────────────────────────────────────────────────────
# 2.  Nested LOOCV – Lasso
# ─────────────────────────────────────────────────────────────────────────────

def run_nested_loocv_lasso(X, y, feature_names, alphas):
    """
    Outer LOOCV loop.  Progress is printed to the screen; results returned.

    Returns
    -------
    y_true      : (n,)    true target values in outer-fold order
    y_pred      : (n,)    predicted values
    best_alphas : (n,)    alpha chosen by inner CV per fold
    coef_matrix : (n, p)  coefficient vector from the final model per fold
    """
    outer_loo = LeaveOneOut()
    splits    = list(outer_loo.split(X))
    n_outer   = len(splits)
    n_feat    = X.shape[1]

    y_true_all  = np.zeros(n_outer)
    y_pred_all  = np.zeros(n_outer)
    best_alphas = np.zeros(n_outer)
    coef_matrix = np.zeros((n_outer, n_feat))

    # ── progress header (screen only) ────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print(f"  Lasso LOOCV:  {n_outer} outer folds", flush=True)
    print(f"{'─'*60}", flush=True)

    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):

        print(f"  Outer fold {fold_i:>{len(str(n_outer))}d} / {n_outer}"
              f"  (test idx: {test_idx[0]})", end="  ", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_train)
        X_te_s  = scaler.transform(X_test)

        inner_model = LassoCV(alphas=alphas, cv=LeaveOneOut(), max_iter=10_000)
        inner_model.fit(X_tr_s, y_train)
        best_alpha = inner_model.alpha_

        final_model = Lasso(alpha=best_alpha, max_iter=10_000)
        final_model.fit(X_tr_s, y_train)
        y_pred = final_model.predict(X_te_s)

        y_true_all[fold_i - 1]  = y_test[0]
        y_pred_all[fold_i - 1]  = y_pred[0]
        best_alphas[fold_i - 1] = best_alpha
        coef_matrix[fold_i - 1] = final_model.coef_

        print(f"-> alpha={best_alpha:.6f}  y_hat={y_pred[0]:+.4f}  y={y_test[0]:+.4f}",
              flush=True)

    print(f"{'─'*60}", flush=True)
    return y_true_all, y_pred_all, best_alphas, coef_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Performance metrics  +  bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(yt, yp):
    return np.sqrt(np.mean((yt - yp) ** 2))

def _mae(yt, yp):
    return np.mean(np.abs(yt - yp))

def _r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _pearson(yt, yp):
    return pearsonr(yt, yp).statistic

def _kendall(yt, yp):
    return kendalltau(yt, yp).statistic

METRICS = {
    "RMSE"      : _rmse,
    "MAE"       : _mae,
    "R2"        : _r2,
    "Pearson r" : _pearson,
    "Kendall t" : _kendall,
}


def bootstrap_ci(y_true, y_pred,
                 n_resamples: int   = 10_000,
                 confidence_level: float = 0.95,
                 random_state: int  = 0) -> dict:
    """
    Paired bootstrap CIs for all five metrics.
    'paired=True' resamples (y_true[i], y_pred[i]) jointly to preserve
    the per-observation correspondence.
    """
    results = {}
    data    = (y_true, y_pred)

    for name, fn in METRICS.items():
        point = fn(y_true, y_pred)
        res   = bootstrap(
            data,
            statistic        = fn,
            n_resamples      = n_resamples,
            confidence_level = confidence_level,
            paired           = True,
            method           = "percentile",
            random_state     = random_state,
        )
        results[name] = {
            "point"  : point,
            "ci_low" : res.confidence_interval.low,
            "ci_high": res.confidence_interval.high,
        }
    return results


def write_performance(out,
                      y_true, y_pred,
                      label            = "Performance",
                      n_resamples: int = 10_000,
                      confidence_level = 0.95):
    """
    Compute metrics + bootstrap CIs and write them to *out* (file).
    A brief notice is printed to the screen so the user knows it is running.
    """
    out.blank()
    out.section(label, wide=True)
    out.write(f"  Bootstrap CIs: {n_resamples:,} resamples  |  "
              f"{confidence_level*100:.0f} % confidence level")
    out.rule(wide=False)
    out.write(f"  {'Metric':<12}  {'Point est.':>12}  {'CI low':>12}  {'CI high':>12}")
    out.write(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

    print(f"  [bootstrapping: '{label}' – please wait ...]", flush=True)
    ci = bootstrap_ci(y_true, y_pred, n_resamples=n_resamples,
                      confidence_level=confidence_level)

    for name, vals in ci.items():
        out.write(f"  {name:<12}  {vals['point']:>12.4f}  "
                  f"{vals['ci_low']:>12.4f}  {vals['ci_high']:>12.4f}")

    out.rule(wide=True)
    return {k: v["point"] for k, v in ci.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-sample summary table
# ─────────────────────────────────────────────────────────────────────────────

def write_sample_table(out, y_true, y_pred, best_alphas):
    df = pd.DataFrame({
        "y_true"     : np.round(y_true,         4),
        "y_pred"     : np.round(y_pred,          4),
        "residual"   : np.round(y_true - y_pred, 4),
        "best_alpha" : np.round(best_alphas,     6),
    })
    out.blank()
    out.write("  Per-sample predictions (Lasso, all outer folds):")
    out.rule(wide=False)
    out.write(df.to_string(index=True))
    out.rule(wide=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Top-N features by mean |coefficient|
# ─────────────────────────────────────────────────────────────────────────────

def extract_top_features(out, coef_matrix, feature_names, top_N):
    """
    Rank features by mean |Lasso coefficient| across outer folds and write
    the ranked table to *out*.
    """
    n_coef = coef_matrix.shape[1]
    if len(feature_names) != n_coef:
        print(f"[warning] feature_names length ({len(feature_names)}) != "
              f"coef columns ({n_coef}). Auto-generating names.", flush=True)
        feature_names = [f"feature_{i:02d}" for i in range(n_coef)]

    mean_abs_coef = np.mean(np.abs(coef_matrix), axis=0)

    feat_df = pd.DataFrame({
        "feature"       : feature_names,
        "mean_abs_coef" : mean_abs_coef,
        "mean_coef"     : np.mean(coef_matrix, axis=0),
        "std_coef"      : np.std(coef_matrix,  axis=0),
    }).sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)

    top_df  = feat_df.head(top_N)
    top_idx = [feature_names.index(f) for f in top_df["feature"]]

    out.blank()
    out.write(f"  Top-{top_N} features by mean |Lasso coefficient| across outer folds")
    out.rule(wide=False)
    out.write(top_df.to_string(index=True))
    out.rule(wide=False)

    return top_idx, top_df, feature_names   # feature_names may have been regenerated


# ─────────────────────────────────────────────────────────────────────────────
# 6.  OLS follow-up on top-N features
# ─────────────────────────────────────────────────────────────────────────────

def run_ols_top_features(X, y, top_idx, top_feature_names):
    """
    Nested LOOCV for plain OLS on the selected features.
    Progress is printed to the screen; results + coefficients returned.

    Returns
    -------
    y_true_ols  : (n,)
    y_pred_ols  : (n,)
    ols_coef_df : DataFrame ['feature', 'ols_coef'] – fit on full dataset
    """
    X_top     = X[:, top_idx]
    outer_loo = LeaveOneOut()
    splits    = list(outer_loo.split(X_top))
    n_outer   = len(splits)

    y_true_all = np.zeros(n_outer)
    y_pred_all = np.zeros(n_outer)

    print(f"\n{'─'*60}", flush=True)
    print(f"  OLS LOOCV on top-{len(top_idx)} features  ({n_outer} folds)",
          flush=True)
    print(f"{'─'*60}", flush=True)

    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"  OLS fold {fold_i:>{len(str(n_outer))}d} / {n_outer}"
              f"  (test idx: {test_idx[0]})", end="  ", flush=True)

        X_train, X_test = X_top[train_idx], X_top[test_idx]
        y_train, y_test = y[train_idx],     y[test_idx]

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_train)
        X_te_s  = scaler.transform(X_test)

        ols    = LinearRegression()
        ols.fit(X_tr_s, y_train)
        y_pred = ols.predict(X_te_s)

        y_true_all[fold_i - 1] = y_test[0]
        y_pred_all[fold_i - 1] = y_pred[0]

        print(f"-> y_hat={y_pred[0]:+.4f}  y={y_test[0]:+.4f}", flush=True)

    print(f"{'─'*60}", flush=True)

    # ── OLS coefficients on full dataset (interpretability + plot) ───────
    scaler_full = StandardScaler()
    X_top_s     = scaler_full.fit_transform(X_top)
    ols_full    = LinearRegression()
    ols_full.fit(X_top_s, y)

    ols_coef_df = pd.DataFrame({
        "feature"  : top_feature_names,
        "ols_coef" : ols_full.coef_,
    }).sort_values("ols_coef", key=np.abs, ascending=False).reset_index(drop=True)

    return y_true_all, y_pred_all, ols_coef_df


def write_ols_coef_table(out, ols_coef_df):
    out.blank()
    out.write("  OLS coefficients (fit on full dataset, top features):")
    out.rule(wide=False)
    out.write(ols_coef_df.to_string(index=True))
    out.rule(wide=False)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Side-by-side comparison table
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison(out, pts_lasso, pts_ols, n_features, top_N):
    col_w          = 10
    header_metrics = list(pts_lasso.keys())

    out.blank()
    out.section("Side-by-side comparison  (point estimates only)", wide=True)

    header = f"  {'Model':<38}"
    for m in header_metrics:
        header += f"  {m:>{col_w}}"
    out.write(header)

    divider = f"  {'─'*38}"
    for _ in header_metrics:
        divider += f"  {'─'*col_w}"
    out.write(divider)

    for row_label, pts in [
        (f"Lasso  (all {n_features} features)", pts_lasso),
        (f"OLS    (top-{top_N} features)",      pts_ols),
    ]:
        row = f"  {row_label:<38}"
        for m in header_metrics:
            row += f"  {pts[m]:>{col_w}.4f}"
        out.write(row)

    out.rule(wide=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Bar chart of OLS coefficients
# ─────────────────────────────────────────────────────────────────────────────

def plot_ols_coefficients(ols_coef_df, plot_path="ols_coefficients.png"):
    """
    Horizontal bar chart of OLS coefficients for the top features.

    Bar length = |coefficient| (absolute value)
    Colour     = green (#2ca02c) if coefficient > 0
                 red   (#d62728) if coefficient < 0
    Features sorted by |coefficient|, largest bar at the top.
    Signed coefficient value annotated at the right of each bar.
    """
    # sort ascending by |coef| so the largest ends up at the top of the chart
    df       = ols_coef_df.copy().sort_values(
                   "ols_coef", key=np.abs, ascending=True
               ).reset_index(drop=True)

    coefs    = df["ols_coef"].values
    features = df["feature"].values
    abs_vals = np.abs(coefs)
    colours  = ["#2ca02c" if c > 0 else "#d62728" for c in coefs]

    fig, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.55)))

    bars = ax.barh(
        y         = features,
        width     = abs_vals,
        color     = colours,
        edgecolor = "white",
        linewidth = 0.6,
        height    = 0.65,
    )

    # annotate signed value at the end of each bar
    x_max = abs_vals.max() if abs_vals.max() > 0 else 1.0
    for bar, coef in zip(bars, coefs):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{coef:+.4f}",
            va="center", ha="left",
            fontsize=8.5, color="#333333",
        )

    # legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#2ca02c", label="Positive coefficient"),
            Patch(facecolor="#d62728", label="Negative coefficient"),
        ],
        loc="lower right", fontsize=9, framealpha=0.85,
    )

    ax.set_xlabel("|OLS coefficient|  (standardized features)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(
        "OLS Regression Coefficients\n"
        "(top Lasso-selected features, fit on full dataset)",
        fontsize=12, pad=12,
    )
    ax.set_xlim(0, x_max * 1.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [plot]  Saved -> {plot_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Data I/O ──────────────────────────────────────────────────────────

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Read data filefrom the command line.')
    parser.add_argument('--data_feat', type=str,   help='Path to the feature data file')
    parser.add_argument('--data_obs',  type=str,   help='Path to the observed data file')
    parser.add_argument('--top_N',     type=int,   help='Number of top-N features to consider')
    args = parser.parse_args()
    
    X, y, sample_ids, feature_names = load_data(args.data_feat, args.data_obs)

    n = len(sample_ids)         # number of data instances or samples
    p = len(feature_names)      # number of features
    top_N = min(args.top_N, p)  # number of top features to use for OLS regression  
    
    #feature_names = [f"feature_{i:02d}" for i in range(p)]
    # In case you want the feature names substituted numerically
    
    #alphas = np.logspace(-4, 1, 60)            # candidate alpha grid
    alphas = np.logspace(-5, 0, 50) # candidate alpha grid, hardcoded!
    
    output_txt  = "results_mdfitml.txt"
    output_plot = "plot_mdfitml.png"

    # ── screen header ────────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("  Nested LOOCV – Lasso Regression", flush=True)
    print(f"  {n} samples  |  {p} features  |  {len(alphas)} candidate alphas",
          flush=True)
    print(f"  Results -> {output_txt}   Plot -> {output_plot}", flush=True)
    print("="*60, flush=True)


    # ── 2. Nested LOOCV – Lasso  (fold progress -> screen) ───────────────
    y_true, y_pred, best_alphas, coef_matrix = run_nested_loocv_lasso(
        X, y, feature_names, alphas
    )


    # ── open output file; everything below goes to the file ───────────────
    with Writer(output_txt) as out:

        out.section("Nested LOOCV – Lasso Regression", wide=True)
        out.write(f"  Samples  : {n}")
        out.write(f"  Features : {p}")
        out.write(f"  Alphas   : {len(alphas)} candidates"
                  f"  [{alphas[0]:.2e} ... {alphas[-1]:.2e}]")
        out.rule(wide=True)


        # ── 3. Lasso performance + bootstrap CIs ─────────────────────────
        pts_lasso = write_performance(
            out, y_true, y_pred,
            label="Lasso performance (all features, nested LOOCV)"
        )


        # ── 4. Per-sample table ──────────────────────────────────────────
        write_sample_table(out, y_true, y_pred, best_alphas)


        # ── 5. Top-N features ────────────────────────────────────────────
        top_idx, top_df, feature_names = extract_top_features(
            out, coef_matrix, feature_names, top_N=top_N
        )
        top_feat_names = list(top_df["feature"])


        # ── 6a. OLS LOOCV  (fold progress -> screen) ──────────────────────
        y_true_ols, y_pred_ols, ols_coef_df = run_ols_top_features(
            X, y, top_idx, top_feat_names
        )
        write_ols_coef_table(out, ols_coef_df)


        # ── 6b. OLS performance + bootstrap CIs ──────────────────────────
        pts_ols = write_performance(
            out, y_true_ols, y_pred_ols,
            label=f"OLS performance (top-{top_N} Lasso features, nested LOOCV)"
        )


        # ── 7. Side-by-side comparison ───────────────────────────────────
        write_comparison(out, pts_lasso, pts_ols,
                         n_features=p, top_N=top_N)


    print(f"\n  [output] Results saved -> {output_txt}", flush=True)


    # ── 8. OLS coefficient bar chart ──────────────────────────────────────
    plot_ols_coefficients(ols_coef_df, plot_path=output_plot)


    print("\n  Done.\n", flush=True)


if __name__ == "__main__":
    main()
