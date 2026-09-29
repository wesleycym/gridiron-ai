# Run: python ml/src/scripts/train_quantiles.py
# This script predicts a range of fantasy outcomes

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore", message=".*eval_set.*")
pl.Config.set_tbl_rows(20)

ML_ROOT = Path(__file__).resolve().parents[2]  # scripts -> src -> ml
DATA_PATH = ML_ROOT / "data" / "processed" / "training.parquet"
OUT_DIR = ML_ROOT / "models" / "quantiles"

TARGET = "y_fantasy_points_ppr"
BASELINE_COL = "fantasy_points_ppr_avg5"
POSITIONS = ["QB", "RB", "WR", "TE"]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
THRESHOLDS = [10, 15, 20, 25]  # "chance of X+ points" that we check for calibration


# Features (same rule as train.py / build_dataset.py)
def feature_columns(df: pl.DataFrame) -> list[str]:
    """Shared with build_dataset.py so the feature list is defined in one place."""
    from build_dataset import feature_columns as shared
    return shared(df)


def to_pandas_features(df: pl.DataFrame, features: list[str]):
    X = df.select(features).to_pandas()
    # Fixed category list, so training and prediction always encode positions the same way
    X["position"] = pd.Categorical(df["position"].to_list(), categories=POSITIONS)
    return X


# --------------------------------------------------------------------------------------
# Turning quantile predictions into probabilities (the backend will reuse this)
# --------------------------------------------------------------------------------------
def fix_crossing(preds: np.ndarray) -> np.ndarray:
    """Separate models can occasionally predict q25 > q50. Sorting each row fixes that."""
    return np.sort(preds, axis=1)


def _full_quantile_curve(preds: np.ndarray, quantiles=QUANTILES):
    """Predicted quantiles plus linearly extended tails at probability 0 and 1."""
    qs = np.array(quantiles)
    lo_gap = preds[:, 1] - preds[:, 0]
    hi_gap = preds[:, -1] - preds[:, -2]
    q0 = preds[:, 0] - 2 * lo_gap - 1e-6
    q1 = preds[:, -1] + 3 * hi_gap + 1e-6
    return np.column_stack([q0, preds, q1]), np.concatenate([[0.0], qs, [1.0]])


def mean_from_quantiles(preds: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    """
    Average expected score, from the predicted distribution.
    Fantasy scores are lopsided (big games stretch the top end), so the mean usually
    sits above the median. Most sites (ESPN, Yahoo) publish means.
    """
    vals, qs = _full_quantile_curve(preds, quantiles)
    # Area under the quantile curve = the mean (trapezoid rule between known points)
    return np.sum((vals[:, 1:] + vals[:, :-1]) / 2 * np.diff(qs)[None, :], axis=1)


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
def apply_offsets(preds: np.ndarray, positions, offsets: dict) -> np.ndarray:
    """Shift each quantile prediction by its calibration offset for that position."""
    out = preds.copy()
    positions = np.asarray(positions)
    for pos, off in offsets.items():
        mask = positions == pos
        out[mask] += np.array(off)[None, :]
    return fix_crossing(out)


def fit_offsets(y_cal: np.ndarray, preds_cal: np.ndarray, positions, min_rows: int = 150) -> dict:
    """
    Conformal calibration. On a season the models were NOT fit on, find how far each
    quantile prediction needs to move so that, e.g., exactly 10% of scores land below q10.
    Done per position, since QBs and TEs miss in different ways.
    """
    positions = np.asarray(positions)
    resid = y_cal[:, None] - preds_cal
    global_off = [float(np.quantile(resid[:, i], q)) for i, q in enumerate(QUANTILES)]
    offsets = {}
    for pos in POSITIONS:
        mask = positions == pos
        if mask.sum() >= min_rows:
            offsets[pos] = [float(np.quantile(resid[mask, i], q)) for i, q in enumerate(QUANTILES)]
        else:
            offsets[pos] = global_off
    return offsets


def fit_level(y_cal: np.ndarray, preds_cal: np.ndarray, positions, min_rows: int = 150) -> dict:
    """
    Fix the "pull toward the middle": on the calibration season, fit
        actual = a + b * projected_mean      (per position)
    b > 1 means the model's projections are too squeezed together, so we stretch them:
    high projections move up, low projections move down.
    """
    positions = np.asarray(positions)
    m = mean_from_quantiles(preds_cal)
    b_all, a_all = np.polyfit(m, y_cal, 1)
    fit = {}
    for pos in POSITIONS:
        mask = positions == pos
        if mask.sum() >= min_rows:
            b, a = np.polyfit(m[mask], y_cal[mask], 1)
            fit[pos] = [float(a), float(b)]
        else:
            fit[pos] = [float(a_all), float(b_all)]
    return fit


def apply_level(preds: np.ndarray, positions, level: dict) -> np.ndarray:
    """Shift each player's whole range so his mean lands where fit_level says it should."""
    if not level:
        return preds
    out = preds.copy()
    positions = np.asarray(positions)
    m = mean_from_quantiles(preds)
    for pos, (a, b) in level.items():
        mask = positions == pos
        target = a + b * m[mask]
        out[mask] += (target - m[mask])[:, None]
    return out


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
    parser.add_argument("--level-fix", action="store_true",
                        help="experimental: stretch projections toward the extremes. Off by "
                             "default: tested on 2025, it slightly hurt accuracy and calibration")
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
    X_cal, y_cal = X_train[~fit_mask], y_train[~fit_mask]
    cols, cal_cols = [], []
    for q in QUANTILES:
        print(f"Training q{int(q * 100):02d}...", end=" ", flush=True)
        m = train_one(q, X_train[fit_mask], y_train[fit_mask], X_cal, y_cal)
        cols.append(m.predict(X_test))
        if len(y_cal):
            cal_cols.append(m.predict(X_cal))
        m.booster_.save_model(str(OUT_DIR / f"lgbm_q{int(q * 100):02d}.txt"))
        print(f"{m.booster_.num_trees()} trees")
    raw_preds = fix_crossing(np.column_stack(cols))

    # ---- Calibrate on the last training season (which the models weren't fit on) -------
    if len(y_cal):
        cal_positions = train.filter(pl.Series(~fit_mask))["position"].to_numpy()
        cal_raw = fix_crossing(np.column_stack(cal_cols))
        offsets = fit_offsets(y_cal, cal_raw, cal_positions)
        level_fit = (fit_level(y_cal, apply_offsets(cal_raw, cal_positions, offsets), cal_positions)
                     if args.level_fix else {})
    else:
        offsets = {p: [0.0] * len(QUANTILES) for p in POSITIONS}
        level_fit = {}
    test_positions = test["position"].to_numpy()
    offset_preds = apply_offsets(raw_preds, test_positions, offsets)       # ranges calibrated
    model_preds = apply_level(offset_preds, test_positions, level_fit)     # + level stretched

    if level_fit:
        print("\nLevel fix learned from the calibration season (actual = a + b * projected):")
    for pos, (a, b) in level_fit.items():
        print(f"  {pos}: b = {b:.2f}  ({'stretches' if b > 1 else 'squeezes'} projections), a = {a:+.2f}")

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
    print(f"  {'':>4}  {'target':>6}  {'before calib':>12}  {'after calib':>11}")
    for i, q in enumerate(QUANTILES):
        before = float(np.mean(y_test <= raw_preds[:, i]))
        after = float(np.mean(y_test <= model_preds[:, i]))
        flag = "" if abs(after - q) < 0.03 else "  <- off"
        print(f"  q{int(q * 100):02d}  {q:>6.0%}  {before:>12.1%}  {after:>11.1%}{flag}")
    for label, p in [("before", raw_preds), ("after", model_preds)]:
        inside = np.mean((y_test >= p[:, 1]) & (y_test <= p[:, 5]))
        print(f"  80% range contained the actual score {inside:.1%} of the time ({label} calibration)")

    print("\nCoverage by position (after calibration): q10 / q50 / q90, targets 10% / 50% / 90%")
    for pos in POSITIONS:
        mk = test_positions == pos
        if mk.sum():
            c = [float(np.mean(y_test[mk] <= model_preds[mk, i])) for i in (1, 3, 5)]
            print(f"  {pos}: {c[0]:5.1%} / {c[1]:5.1%} / {c[2]:5.1%}   ({mk.sum():,} rows)")

    # ---- Level check: are high projections systematically too low (or too high)? -------
    # Group players by the model's mean projection, then compare with what they scored.
    # If the "20+" group actually averages 23, the model is shrinking stars toward the middle.
    bins = [5, 10, 15, 20]
    labels = ["<5", "5-10", "10-15", "15-20", "20+"]

    def level_frame(preds: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "position": test_positions, "projected": mean_from_quantiles(preds), "actual": y_test,
        }).with_columns(pl.col("projected").cut(bins, labels=labels).alias("proj_bucket"))

    level_before, level = level_frame(offset_preds), level_frame(model_preds)

    def level_table(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by("proj_bucket")
            .agg(
                pl.len().alias("rows"),
                pl.col("projected").mean().round(1).alias("avg_projected"),
                pl.col("actual").mean().round(1).alias("avg_actual"),
            )
            .with_columns((pl.col("avg_actual") - pl.col("avg_projected")).round(1).alias("actual_minus_proj"))
            .sort("avg_projected")
        )

    print("\nLevel check: mean projection vs actual average, by projection size")
    print("(actual_minus_proj should be near 0 in every row; a growing positive number")
    print(" in the top rows means the model undersells high-end players)")
    rel = base_test >= 8
    if level_fit:
        print("\nAll positions, BEFORE level fix")
        print(level_table(level_before))
        print("\nAll positions, AFTER level fix")
        print(level_table(level))
        mae_b = float(np.mean(np.abs(y_test[rel] - level_before["projected"].to_numpy()[rel])))
        mae_a = float(np.mean(np.abs(y_test[rel] - level["projected"].to_numpy()[rel])))
        print(f"\nMean projection error, fantasy-relevant players: {mae_b:.2f} before -> {mae_a:.2f} after")
    else:
        print("\nAll positions")
        print(level_table(level))
        mae = float(np.mean(np.abs(y_test[rel] - level["projected"].to_numpy()[rel])))
        print(f"\nMean projection error, fantasy-relevant players: {mae:.2f}")
    print("\nBy position:")
    for pos in POSITIONS:
        sub = level.filter(pl.col("position") == pos)
        if sub.height:
            print(f"\n{pos}")
            print(level_table(sub))

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
        "positions": POSITIONS,
        "offsets": offsets,
        "level": level_fit,
        "features": list(X_train.columns),
        "test_season": test_season,
        "avg_pinball_model": tot_m / len(QUANTILES),
        "avg_pinball_baseline": tot_b / len(QUANTILES),
    }, indent=2))
    print(f"\nSaved {len(QUANTILES)} models -> {OUT_DIR.relative_to(ML_ROOT)}")


if __name__ == "__main__":
    main()