# Install dependencies: pip install lightgbm scikit-learn polars pandas pyarrow
# Run: python ml/src/scripts/train.py

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import polars as pl
from sklearn.metrics import mean_absolute_error

# Newer LightGBM versions warn about eval_set
warnings.filterwarnings("ignore", message=".*eval_set.*")
pl.Config.set_tbl_rows(20)

ML_ROOT = Path(__file__).resolve().parents[2]  # scripts -> src -> ml
DATA_PATH = ML_ROOT / "data" / "processed" / "training.parquet"
MODEL_DIR = ML_ROOT / "models"

TARGET = "y_fantasy_points_ppr"
BASELINE_COL = "fantasy_points_ppr_avg5"  # "just use his last-5-game average"


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Same rule build_dataset.py uses to name features."""
    return [
        c for c in df.columns
        if c.endswith(("_avg3", "_avg5", "_szn", "_sd5"))
        or c.startswith(("prior_games", "opp_"))
    ]


def to_pandas_features(df: pl.DataFrame, features: list[str]):
    X = df.select(features).to_pandas()
    # Position as a category so the model can learn QB/RB/WR/TE differences
    X["position"] = df["position"].to_pandas().astype("category")
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

    # Default test season = the latest season that's finished
    test_season = args.test_season or (seasons[-2] if len(seasons) > 2 else seasons[-1])

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
    print("Fantasy-relevant players (last-5 avg >= 8 pts):")
    print(f"  rows: {relevant.height}")
    print(f"  baseline MAE: {(relevant['actual'] - relevant['baseline']).abs().mean():.2f}")
    print(f"  model MAE:    {(relevant['actual'] - relevant['model']).abs().mean():.2f}")

    print("\n")

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