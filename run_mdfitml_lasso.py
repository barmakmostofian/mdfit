"""
nested_loocv_lasso.py
=====================
Nested Leave-One-Out Cross-Validation with Lasso regression.

Structure
---------
  1. Outer LOOCV loop  (progress-printed, fold i of N)
       └─ Inner LOOCV via LassoCV  →  selects best alpha per fold
       └─ Final Lasso refit with best alpha  →  prediction + coefficient harvest
  2. Aggregate outer-loop performance  (RMSE, MAE, R²)
  3. Per-sample summary table
  4. Top-10 features by mean |coefficient| across all outer folds
  5. OLS follow-up trained & evaluated (nested LOOCV) on the top-10 features only

Replace X and y at the top of main() with your own data.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



# ─────────────────────────────────────────────────────────────────────────────
# 0. Load Data
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
    
    X = np.array(df_data.drop('potency', axis=1))
    y = np.array(df_data['potency'])
    sample_ids = np.array(df_data.index)
    feature_names = list(df_data.columns)

    return (X, y, sample_ids, feature_names)



# ─────────────────────────────────────────────────────────────────────────────
# 1.  Nested LOOCV – Lasso
# ─────────────────────────────────────────────────────────────────────────────

def run_nested_loocv_lasso(X, y, feature_numbers, alphas):
    """
    Outer LOOCV loop.

    For every outer fold:
      • inner LassoCV (LOOCV) selects the best alpha on the training set
      • a final Lasso is refit with that alpha  →  used for prediction AND
        coefficient collection

    Returns
    -------
    y_true        : (n,)  true target values in outer-fold order
    y_pred        : (n,)  predicted values
    best_alphas   : (n,)  alpha chosen by inner CV per fold
    coef_matrix   : (n, p)  coefficient vector from the final model per fold
    """

    outer_loo = LeaveOneOut()
    splits    = list(outer_loo.split(X))
    n_outer   = len(splits)
    n_feat    = X.shape[1]

    y_true_all   = np.zeros(n_outer)
    y_pred_all   = np.zeros(n_outer)
    best_alphas  = np.zeros(n_outer)
    coef_matrix  = np.zeros((n_outer, n_feat))   # one coef vector per outer fold

    print(f"\n{'─'*60}")
    print(f"  Outer LOOCV:  {n_outer} folds total")
    print(f"{'─'*60}")

    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):

        # ── progress counter ────────────────────────────────────────────
        print(f"  Outer fold {fold_i:>{len(str(n_outer))}d} / {n_outer} "
              f"  (test sample index: {test_idx[0]})", end="  ")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── scale on train only (no leakage) ────────────────────────────
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_train)
        X_te_s   = scaler.transform(X_test)

        # ── inner LOOCV: find best alpha ─────────────────────────────────
        inner_model = LassoCV(
            alphas   = alphas,
            cv       = LeaveOneOut(),
            max_iter = 10_000,
        )
        inner_model.fit(X_tr_s, y_train)
        best_alpha = inner_model.alpha_

        # ── refit final model with chosen alpha ──────────────────────────
        final_model = Lasso(alpha=best_alpha, max_iter=10_000)
        final_model.fit(X_tr_s, y_train)

        y_pred = final_model.predict(X_te_s)

        # store results
        y_true_all[fold_i - 1]  = y_test[0]
        y_pred_all[fold_i - 1]  = y_pred[0]
        best_alphas[fold_i - 1] = best_alpha
        coef_matrix[fold_i - 1] = final_model.coef_

        print(f"→  best α = {best_alpha:.6f}  |  "
              f"ŷ = {y_pred[0]:+.4f}  (y = {y_test[0]:+.4f})")

    print(f"{'─'*60}")
    return y_true_all, y_pred_all, best_alphas, coef_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def print_performance(y_true, y_pred, label="Nested LOOCV – Lasso"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"{'═'*60}")
    return rmse, mae, r2


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-sample summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_sample_table(y_true, y_pred, best_alphas):
    df = pd.DataFrame({
        "y_true"     : np.round(y_true, 4),
        "y_pred"     : np.round(y_pred, 4),
        "residual"   : np.round(y_true - y_pred, 4),
        "best_alpha" : np.round(best_alphas, 6),
    })
    print("\n  Per-sample results (all rows):")
    print(df.to_string(index=True))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Top-10 features by mean |coefficient|
# ─────────────────────────────────────────────────────────────────────────────

def extract_top_features(coef_matrix, feature_numbers, top_n=10):
    """
    Average the absolute coefficient across all outer folds,
    then rank features and return the top-N indices + a display table.

    Using mean |coef| rather than a single-fold coefficient gives a more
    stable importance estimate because it averages over the whole dataset.
    """
    mean_abs_coef = np.mean(np.abs(coef_matrix), axis=0)   # shape (p,)

    feat_df = pd.DataFrame({
        "feature"       : feature_numbers,
        "mean_abs_coef" : mean_abs_coef,
        "mean_coef"     : np.mean(coef_matrix, axis=0),
        "std_coef"      : np.std(coef_matrix, axis=0),
    }).sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)

    top_df  = feat_df.head(top_n)
    top_idx = [feature_numbers.index(f) for f in top_df["feature"]]

    print(f"\n{'─'*60}")
    print(f"  Top-{top_n} features by mean |Lasso coefficient| across outer folds")
    print(f"{'─'*60}")
    print(top_df.to_string(index=True))
    print(f"{'─'*60}")

    return top_idx, top_df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OLS follow-up on top-10 features only  (also nested LOOCV)
# ─────────────────────────────────────────────────────────────────────────────

def run_ols_top_features(X, y, top_idx, top_feature_numbers):
    """
    Train a plain OLS (no regularisation) on the top-N Lasso features,
    evaluated with the same outer LOOCV so results are directly comparable.
    """
    X_top = X[:, top_idx]

    outer_loo = LeaveOneOut()
    splits    = list(outer_loo.split(X_top))
    n_outer   = len(splits)

    y_true_all = np.zeros(n_outer)
    y_pred_all = np.zeros(n_outer)

    print(f"\n{'─'*60}")
    print(f"  OLS follow-up on top-{len(top_idx)} features  "
          f"({n_outer} outer LOOCV folds)")
    print(f"{'─'*60}")

    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):

        print(f"  OLS fold {fold_i:>{len(str(n_outer))}d} / {n_outer} "
              f"  (test index: {test_idx[0]})", end="  ")

        X_train, X_test = X_top[train_idx], X_top[test_idx]
        y_train, y_test = y[train_idx],     y[test_idx]

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_train)
        X_te_s   = scaler.transform(X_test)

        ols = LinearRegression()
        ols.fit(X_tr_s, y_train)
        y_pred = ols.predict(X_te_s)

        y_true_all[fold_i - 1] = y_test[0]
        y_pred_all[fold_i - 1] = y_pred[0]

        print(f"→  ŷ = {y_pred[0]:+.4f}  (y = {y_test[0]:+.4f})")

    print(f"{'─'*60}")

    # ── OLS coefficients on full dataset (for interpretability) ─────────
    scaler_full = StandardScaler()
    X_top_s     = scaler_full.fit_transform(X_top)
    ols_full    = LinearRegression()
    ols_full.fit(X_top_s, y)

    ols_coef_df = pd.DataFrame({
        "feature"    : top_feature_numbers,
        "ols_coef"   : ols_full.coef_,
    }).sort_values("ols_coef", key=np.abs, ascending=False).reset_index(drop=True)

    print("\n  OLS coefficients (fit on full dataset, top features):")
    print(ols_coef_df.to_string(index=True))

    return y_true_all, y_pred_all


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── 0. Load data ──────────────────────────────────────────────────────────

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Read data filefrom the command line.')
    parser.add_argument('--data_feat', type=str,   help='Path to the feature data file')
    parser.add_argument('--data_obs',  type=str,   help='Path to the observed data file')
    args = parser.parse_args()

    X, y, sample_ids, feature_names = load_data(args.data_feat, args.data_obs) 
  
    n = X.shape[0]            # number of data instances or samples
    p = X.shape[1]            # number of features

    feature_numbers = [f"feature_{i:02d}" for i in range(p)]

    alphas = np.logspace(-4, 1, 60) # candidate alpha grid

    print("\n" + "═"*60)
    print("  Nested LOOCV – Lasso Regression")
    print(f"  {n} samples  |  {p} features  |  {len(alphas)} candidate alphas")
    print("═"*60)


    # ── 1. Nested LOOCV – Lasso ──────────────────────────────────────────

    y_true, y_pred, best_alphas, coef_matrix = run_nested_loocv_lasso(
        X, y, feature_numbers, alphas
    )



    # ── 2. Aggregate performance ─────────────────────────────────────────

    print_performance(y_true, y_pred, label="Nested LOOCV – Lasso  (all features)")



    # ── 3. Per-sample table ──────────────────────────────────────────────

    print_sample_table(y_true, y_pred, best_alphas)



    # ── 4. Top-10 features ───────────────────────────────────────────────

    top_n           = min(10, p)               # graceful if p < 10
    top_idx, top_df = extract_top_features(coef_matrix, feature_numbers, top_n=top_n)
    top_feat_names  = list(top_df["feature"])



    # ── 5. OLS on top-10 features ────────────────────────────────────────

    y_true_ols, y_pred_ols = run_ols_top_features(X, y, top_idx, top_feat_names)

    print_performance(
        y_true_ols, y_pred_ols,
        label=f"Nested LOOCV – OLS on top-{top_n} Lasso features"
    )



    # ── 6. Side-by-side comparison ───────────────────────────────────────

    rmse_lasso = np.sqrt(mean_squared_error(y_true,     y_pred))
    rmse_ols   = np.sqrt(mean_squared_error(y_true_ols, y_pred_ols))
    r2_lasso   = r2_score(y_true,     y_pred)
    r2_ols     = r2_score(y_true_ols, y_pred_ols)

    print(f"\n{'═'*60}")
    print("  Side-by-side comparison")
    print(f"{'─'*60}")
    print(f"  {'Model':<40} {'RMSE':>8}  {'R²':>8}")
    print(f"  {'─'*40} {'─'*8}  {'─'*8}")
    print(f"  {'Lasso  (all features, nested LOOCV)':<40} "
          f"{rmse_lasso:>8.4f}  {r2_lasso:>8.4f}")
    print(f"  {'OLS    (top-' + str(top_n) + ' features, nested LOOCV)':<40} "
          f"{rmse_ols:>8.4f}  {r2_ols:>8.4f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
