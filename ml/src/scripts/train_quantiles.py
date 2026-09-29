# Run: python ml/src/scripts/train_quantiles.py
# This script predicts a range of fantasy outcomes

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message=".*eval_set.*")
pl.Config.set_tbl_rows(20)

ML_ROOT = Path(__file__).resolve().parents[2]  # scripts -> src -> ml
DATA_PATH = ML_ROOT / "data" / "processed" / "training.parquet"
OUT_DIR = ML_ROOT / "models" / "quantiles"

TARGET = "y_fantasy_points_ppr"
BASELINE_COL = "fantasy_points_ppr_avg5"
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
THRESHOLDS = [10, 15, 20, 25]  # "chance of X+ points" that we check for calibration


# --------------------------------------------------------------------------------------
# Features (same rule as train.py / build_dataset.py)
# --------------------------------------------------------------------------------------
def feature_columns(df: pl.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.endswith(("_avg3", "_avg5", "_szn", "_sd5"))
        or c.startswith(("prior_games", "opp_", "vegas_"))
    ]


def to_pandas_features(df: pl.DataFrame, features: list[str]):
    X = df.select(features).to_pandas()
    X["position"] = df["position"].to_pandas().astype("category")
    return X


# --------------------------------------------------------------------------------------
# Turning quantile predictions into probabilities (the backend will reuse this)
# --------------------------------------------------------------------------------------
def fix_crossing(preds: np.ndarray) -> np.ndarray:
    """Separate models can occasionally predict q25 > q50. Sorting each row fixes that."""
    return np.sort(preds, axis=1)


def prob_over(preds: np.ndarray, threshold: float, quantiles=QUANTILES) -> np.ndarray:
    """
    P(score >= threshold) for each row, by interpolating between the predicted quantiles.

    preds: shape (n_players, n_quantiles), sorted per row.
    Beyond the outermost quantiles, the tails are extended linearly so that e.g. a backup
    TE's chance of 30+ points can go toward 0 instead of being stuck at 5%.
    """
    qs = np.array(quantiles)
    lo_gap = preds[:, 1] - preds[:, 0]
    hi_gap = preds[:, -1] - preds[:, -2]
    # Anchor points for probability 0 and 1, extending the tails
    q0 = preds[:, 0] - 2 * lo_gap - 1e-6
    q1 = preds[:, -1] + 3 * hi_gap + 1e-6
    full_vals = np.column_stack([q0, preds, q1])
    full_qs = np.concatenate([[0.0], qs, [1.0]])

    cdf = np.array([np.interp(threshold, row, full_qs) for row in full_vals])
    return 1.0 - cdf


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Standard loss for quantile predictions. Lower is better."""
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def train_one(q: float, X_fit, y_fit, X_val, y_val):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
    )
    has_val = len(y_val) > 0
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)] if has_val else None,
        callbacks=[lgb.early_stopping(100, verbose=False)] if has_val else None,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quantile models for PPR points.")
    parser.add_argument("--test-season", type=int, default=None)
    args = parser.parse_args()

    df = pl.read_parquet(DATA_PATH)
    seasons = sorted(df["season"].unique().to_list())
    test_season = args.test_season or (seasons[-2] if len(seasons) > 2 else seasons[-1])

    train = df.filter(pl.col("season") < test_season)
    test = df.filter(pl.col("season") == test_season)
    print(f"Train seasons: {sorted(train['season'].unique().to_list())}  ({train.height:,} rows)")
    print(f"Test season:   {test_season}  ({test.height:,} rows)\n")

    features = feature_columns(df)
    X_train, X_test = to_pandas_features(train, features), to_pandas_features(test, features)
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    # Last training season is used to decide when to stop adding trees
    fit_mask = (train["season"] < train["season"].max()).to_numpy()
    if fit_mask.sum() == 0:
        fit_mask[:] = True

    # ---- Baseline: last-5 average + typical miss sizes from the training data ----------
    # e.g. if players historically land 7 pts below their avg5 10% of the time,
    # the baseline q10 is "avg5 - 7". Same spread for everyone.
    base_train = train[BASELINE_COL].fill_null(0.0).to_numpy()
    base_test = test[BASELINE_COL].fill_null(0.0).to_numpy()
    resid_q = np.quantile(y_train - base_train, QUANTILES)
    baseline_preds = fix_crossing(base_test[:, None] + resid_q[None, :])

    # ---- Train one model per quantile -------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = []
    for q in QUANTILES:
        print(f"Training q{int(q * 100):02d}...", end=" ", flush=True)
        m = train_one(q, X_train[fit_mask], y_train[fit_mask], X_train[~fit_mask], y_train[~fit_mask])
        cols.append(m.predict(X_test))
        m.booster_.save_model(str(OUT_DIR / f"lgbm_q{int(q * 100):02d}.txt"))
        print(f"{m.booster_.num_trees()} trees")
    model_preds = fix_crossing(np.column_stack(cols))

    # ---- 1. Pinball loss: overall quality of the ranges --------------------------------
    print("\nPinball loss by quantile (lower is better)")
    print(f"  {'quantile':>8}  {'baseline':>9}  {'model':>7}")
    tot_b = tot_m = 0.0
    for i, q in enumerate(QUANTILES):
        b = pinball_loss(y_test, baseline_preds[:, i], q)
        m = pinball_loss(y_test, model_preds[:, i], q)
        tot_b, tot_m = tot_b + b, tot_m + m
        print(f"  {q:>8.2f}  {b:>9.3f}  {m:>7.3f}")
    print(f"  {'average':>8}  {tot_b / len(QUANTILES):>9.3f}  {tot_m / len(QUANTILES):>7.3f}"
          f"   ({(tot_b - tot_m) / tot_b * 100:+.1f}%)")

    # ---- 2. Coverage: is q10 actually beaten 90% of the time? --------------------------
    print("\nCoverage: share of actual scores below each predicted quantile (should match)")
    for i, q in enumerate(QUANTILES):
        share = float(np.mean(y_test <= model_preds[:, i]))
        flag = "" if abs(share - q) < 0.03 else "  <- off"
        print(f"  q{int(q * 100):02d}: {share:6.1%}  (target {q:.0%}){flag}")
    inside = np.mean((y_test >= model_preds[:, 1]) & (y_test <= model_preds[:, 5]))
    print(f"  80% range (q10 to q90) contained the actual score {inside:.1%} of the time")

    # ---- 3. Threshold probabilities: when we say 30%, does it happen 30% of the time? --
    print("\nCalibration of 'chance of X+ points' (fantasy-relevant players, avg5 >= 8)")
    relevant = base_test >= 8
    for t in THRESHOLDS:
        p = prob_over(model_preds[relevant], t)
        hit = (y_test[relevant] >= t).astype(float)
        brier = float(np.mean((p - hit) ** 2))
        p_base = prob_over(baseline_preds[relevant], t)
        brier_base = float(np.mean((p_base - hit) ** 2))
        print(f"\n  {t}+ points   Brier: model {brier:.4f} vs baseline {brier_base:.4f} (lower is better)")
        bins = [0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0001]
        tbl = (
            pl.DataFrame({"pred": p, "hit": hit})
            .with_columns(pl.col("pred").cut(bins[1:-1]).alias("bucket"))
            .group_by("bucket")
            .agg(
                pl.len().alias("rows"),
                (pl.col("pred").mean() * 100).round(1).alias("predicted_%"),
                (pl.col("hit").mean() * 100).round(1).alias("actual_%"),
            )
            .filter(pl.col("rows") >= 20)
            .sort("predicted_%")
        )
        print(tbl)

    # ---- Example output ------------------------------------------------------------------
    last_week = test["week"].max()
    ex_mask = (test["week"] == last_week).to_numpy() & relevant
    ex = test.filter(pl.Series(ex_mask)).select("player_display_name", "position", TARGET)
    ex_preds = model_preds[ex_mask]
    ex = ex.with_columns(
        pl.Series("floor_q10", ex_preds[:, 1]).round(1),
        pl.Series("median", ex_preds[:, 3]).round(1),
        pl.Series("ceiling_q90", ex_preds[:, 5]).round(1),
        pl.Series("p_20plus", prob_over(ex_preds, 20) * 100).round(0),
    ).rename({TARGET: "actual"}).with_columns(pl.col("actual").round(1)).sort("median", descending=True).head(12)
    print(f"\nExample: top projected players, week {last_week} of {test_season}")
    print(ex)

    (OUT_DIR / "meta.json").write_text(json.dumps({
        "quantiles": QUANTILES,
        "features": list(X_train.columns),
        "test_season": test_season,
        "avg_pinball_model": tot_m / len(QUANTILES),
        "avg_pinball_baseline": tot_b / len(QUANTILES),
    }, indent=2))
    print(f"\nSaved {len(QUANTILES)} models -> {OUT_DIR.relative_to(ML_ROOT)}")


if __name__ == "__main__":
    main()