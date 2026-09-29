# Install dependencies: pip install lightgbm scikit-learn polars pandas pyarrow
# Run: python ml/src/scripts/train.py

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error

# Newer LightGBM versions warn about eval_set; it still works, and older versions need it.
warnings.filterwarnings("ignore", message=".*eval_set.*")
pl.Config.set_tbl_rows(20)

ML_ROOT = Path(__file__).resolve().parents[2]  # scripts -> src -> ml
DATA_PATH = ML_ROOT / "data" / "processed" / "training.parquet"
MODEL_DIR = ML_ROOT / "models"

TARGET = "y_fantasy_points_ppr"
BASELINE_COL = "fantasy_points_ppr_avg5"  # "just use his last-5-game average"


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Shared with build_dataset.py so the feature list is defined in one place."""
    from build_dataset import feature_columns as shared
    return shared(df)


def to_pandas_features(df: pl.DataFrame, features: list[str]):
    X = df.select(features).to_pandas()
    # Position as a category so the model can learn QB/RB/WR/TE differences
    X["position"] = pd.Categorical(df["position"].to_list(), categories=["QB", "RB", "WR", "TE"])
    return X


def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline + LightGBM on PPR points.")
    parser.add_argument("--test-season", type=int, default=None,
                        help="season held out for testing (default: latest complete season)")
    args = parser.parse_args()

    df = pl.read_parquet(DATA_PATH)
    seasons = sorted(df["season"].unique().to_list())

    # Default test season = the latest season that's finished. The current season
    # is only partly played, so the one before it is a fairer test.
    test_season = args.test_season or (seasons[-2] if len(seasons) > 2 else seasons[-1])

    # Time-based split: train on the past, test on a season the model never saw.
    # Anything after the test season is ignored here so we never train on the future.
    train = df.filter(pl.col("season") < test_season)
    test = df.filter(pl.col("season") == test_season)
    print(f"Train seasons: {sorted(train['season'].unique().to_list())}  ({train.height:,} rows)")
    print(f"Test season:   {test_season}  ({test.height:,} rows)\n")

    features = feature_columns(df)
    X_train, X_test = to_pandas_features(train, features), to_pandas_features(test, features)
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    # ---- Step 1: baseline ------------------------------------------------------
    baseline_pred = test[BASELINE_COL].fill_null(0.0).to_numpy()
    baseline_mae = mae(y_test, baseline_pred)

    # ---- Step 2: LightGBM ------------------------------------------------------
    # Hold out the last training season to decide when to stop adding trees.
    last_train_season = train["season"].max()
    fit_mask = (train["season"] < last_train_season).to_numpy()
    if fit_mask.sum() == 0:  # only one training season available
        fit_mask[:] = True

    model = lgb.LGBMRegressor(
        objective="regression_l1",  # optimize MAE directly; robust to boom-game outliers
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
    )
    model.fit(
        X_train[fit_mask], y_train[fit_mask],
        eval_set=[(X_train[~fit_mask], y_train[~fit_mask])] if (~fit_mask).any() else None,
        callbacks=[lgb.early_stopping(100, verbose=False)] if (~fit_mask).any() else None,
    )
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)

    # ---- Step 3: compare -------------------------------------------------------
    print("Mean absolute error on test season (lower is better)")
    print(f"  Baseline (last-5 avg): {baseline_mae:6.2f} pts")
    print(f"  LightGBM:              {model_mae:6.2f} pts")
    gain = (baseline_mae - model_mae) / baseline_mae * 100
    print(f"  Improvement:           {gain:+6.1f}%\n")

    print("By position:")
    results = test.select("position").with_columns(
        pl.Series("actual", y_test),
        pl.Series("baseline", baseline_pred),
        pl.Series("model", model_pred),
    )
    by_pos = (
        results.group_by("position")
        .agg(
            pl.len().alias("rows"),
            (pl.col("actual") - pl.col("baseline")).abs().mean().round(2).alias("baseline_mae"),
            (pl.col("actual") - pl.col("model")).abs().mean().round(2).alias("model_mae"),
        )
        .sort("position")
    )
    print(by_pos, "\n")

    relevant = results.filter(pl.col("baseline") >= 8)
    rel_base = (relevant["actual"] - relevant["baseline"]).abs().mean()
    rel_model = (relevant["actual"] - relevant["model"]).abs().mean()
    print("Fantasy-relevant players (last-5 avg >= 8 pts):")
    print(f"  rows: {relevant.height:,}")
    print(f"  baseline MAE: {rel_base:.2f}")
    print(f"  model MAE:    {rel_model:.2f}  ({(rel_base - rel_model) / rel_base * 100:+.1f}%)\n")

    print("Top 15 features the model relied on:")
    importance = (
        pl.DataFrame({
            "feature": model.booster_.feature_name(),
            "gain": model.booster_.feature_importance(importance_type="gain"),
        })
        .with_columns((pl.col("gain") / pl.col("gain").sum() * 100).round(1).alias("pct"))
        .sort("gain", descending=True)
        .select("feature", "pct")
        .head(15)
    )
    print(importance, "\n")

    # ---- Save ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / "lgbm_ppr.txt"))
    (MODEL_DIR / "lgbm_ppr_features.json").write_text(
        json.dumps({"features": list(X_train.columns), "test_season": test_season,
                    "test_mae": model_mae, "baseline_mae": baseline_mae}, indent=2)
    )
    print(f"Saved model -> {(MODEL_DIR / 'lgbm_ppr.txt').relative_to(ML_ROOT)}")


if __name__ == "__main__":
    main()