"""
End-to-end workflow for the Kaggle-style car price regression task.

Steps:
1. Load the provided train/test csv files from Downloads.
2. Run light-weight EDA summaries for sanity checks.
3. Apply feature engineering and clean-up helpers shared between models.
4. Train/evaluate multiple tree ensembles (LightGBM, XGBoost, CatBoost).
5. Fit the best model on the full dataset and dump submit.csv.

The script writes:
- reports/eda_summary.md   (textual stats used in the write-up)
- artifacts/model_metrics.json  (cross-val MAE per model)
- submit.csv               (ID,Predict for the hidden test set)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = Path("/Users/aleksey/Downloads/cars_train.csv")
TEST_PATH = Path("/Users/aleksey/Downloads/cars-1-hw/cars_test.csv")

LUXURY_MAKES = {
    "BMW",
    "MERCEDES-BENZ",
    "LEXUS",
    "PORSCHE",
    "AUDI",
    "BENTLEY",
    "JAGUAR",
    "INFINITI",
    "CADILLAC",
    "LAND ROVER",
    "VOLVO",
    "TESLA",
    "MASERATI",
    "LINCOLN",
}
POPULAR_COLORS = {"Black", "White", "Silver", "Gray", "Grey"}
SUV_KEYWORDS = ("SUV", "Jeep", "Crossover", "truck", "Pickup", "Van")
YEAR_BINS = [1985, 1995, 2000, 2005, 2010, 2015, 2018, 2021, 2026]


def _extract_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", np.nan)
    )
    return cleaned.astype(float)


def _prepare_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    result = df.copy()
    result["Engine"] = _extract_number(result["Engine"])
    result["Distance"] = _extract_number(result["Distance"])
    result["Cylinders"] = pd.to_numeric(result["Cylinders"], errors="coerce")

    current_year = max(result["Year"].max(), 2024)
    result["Age"] = (current_year + 1) - result["Year"]
    result["Age"] = result["Age"].clip(lower=0)

    result["Distance_log"] = np.log1p(result["Distance"])
    result["Engine_log"] = np.log1p(result["Engine"])
    result["Distance_per_year"] = result["Distance"] / (result["Age"] + 1)
    result["Engine_per_cyl"] = result["Engine"] / result["Cylinders"].replace(0, np.nan)
    result["Is_new_distance"] = (result["Distance"] < 1000).astype(int)
    result["Is_zero_distance"] = (result["Distance"] == 0).astype(int)

    # Basic binary indicators
    result["Transmission_manual"] = (result["Transmission"] == "Manual").astype(int)
    result["Drive_4x4"] = (result["Drive"] == "4x4").astype(int)
    result["Wheel_right"] = (result["Wheel"] == "Right-hand drive").astype(int)
    result["Is_automatic"] = (result["Transmission"] == "Automatic").astype(int)
    result["Is_diesel"] = result["Fuel"].str.contains("Diesel", case=False, na=False).astype(int)
    result["Is_hybrid"] = result["Fuel"].str.contains("Hybrid|Electric", case=False, regex=True, na=False).astype(int)

    categorical_cols = ["Make", "Model", "Style", "Fuel", "Color", "Transmission", "Drive", "Wheel"]
    for col in categorical_cols:
        result[col] = result[col].fillna("Unknown").astype(str).str.strip()

    result["Make_Model"] = (result["Make"] + "_" + result["Model"]).str.replace(r"\s+", "", regex=True)
    result["Make_Style"] = (result["Make"] + "_" + result["Style"]).str.replace(r"\s+", "", regex=True)
    result["Fuel_Transmission"] = (result["Fuel"] + "_" + result["Transmission"]).str.replace(r"\s+", "", regex=True)
    result["Is_luxury_make"] = result["Make"].str.upper().isin(LUXURY_MAKES).astype(int)
    result["Is_popular_color"] = result["Color"].str.title().isin(POPULAR_COLORS).astype(int)
    result["Is_suv_style"] = result["Style"].str.contains("|".join(SUV_KEYWORDS), case=False, na=False).astype(int)
    result["Year_bucket"] = (
        pd.cut(result["Year"], bins=YEAR_BINS, labels=False, include_lowest=True)
        .fillna(0)
        .astype(int)
    )
    result["Is_modern"] = (result["Year"] >= 2016).astype(int)
    result["Is_oldtimer"] = (result["Year"] <= 2005).astype(int)

    numeric_cols = [
        "Engine",
        "Distance",
        "Cylinders",
        "Age",
        "Distance_log",
        "Engine_log",
        "Distance_per_year",
        "Engine_per_cyl",
    ]
    for col in numeric_cols:
        result[col] = result[col].fillna(result[col].median())

    result["Log_Age"] = np.log1p(result["Age"])
    result["Engine_to_age"] = result["Engine"] / (result["Age"] + 1)
    result["Distance_engine_ratio"] = result["Distance"] / (
        np.power(result["Engine"].fillna(result["Engine"].median()) + 0.1, 1.2)
    )
    result["Cylinder_density"] = result["Cylinders"] / (result["Engine"] + 0.1)
    result["Engine_to_age"] = result["Engine_to_age"].replace([np.inf, -np.inf], np.nan)
    result["Cylinder_density"] = result["Cylinder_density"].replace([np.inf, -np.inf], np.nan)
    result["Distance_engine_ratio"] = result["Distance_engine_ratio"].replace([np.inf, -np.inf], np.nan)
    for col in ["Engine_to_age", "Distance_engine_ratio", "Cylinder_density"]:
        result[col] = result[col].fillna(result[col].median())

    if is_train and result["Price"].isna().any():
        result = result.dropna(subset=["Price"])

    return result


def apply_shared_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    count_cols = [
        "Make",
        "Model",
        "Style",
        "Fuel",
        "Color",
        "Transmission",
        "Drive",
        "Wheel",
        "Make_Model",
        "Make_Style",
        "Fuel_Transmission",
    ]
    for col in count_cols:
        counts = combined[col].value_counts()
        train_df[f"{col}_count"] = train_df[col].map(counts).astype(float)
        test_df[f"{col}_count"] = test_df[col].map(counts).astype(float)

    return train_df, test_df


def add_target_encodings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    columns: List[str],
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = y.mean()

    for col in columns:
        oof = pd.Series(index=train_df.index, dtype=float)
        for train_idx, val_idx in kf.split(train_df):
            mapping = (
                pd.DataFrame({"feature": train_df.iloc[train_idx][col], "target": y.iloc[train_idx]})
                .groupby("feature")["target"]
                .mean()
            )
            oof.iloc[val_idx] = train_df.iloc[val_idx][col].map(mapping)

        train_df[f"{col}_te"] = oof.fillna(global_mean)
        final_mapping = (
            pd.DataFrame({"feature": train_df[col], "target": y})
            .groupby("feature")["target"]
            .mean()
        )
        test_df[f"{col}_te"] = test_df[col].map(final_mapping).fillna(global_mean)

    return train_df, test_df


def generate_eda_report(df: pd.DataFrame) -> None:
    lines: List[str] = []
    price = df["Price"]
    lines.append("# EDA Snapshot")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Price mean/median/std: {price.mean():.0f} / {price.median():.0f} / {price.std():.0f}")
    lines.append(f"- Year span: {df['Year'].min()} — {df['Year'].max()}")
    lines.append(f"- Distance median (km): {df['Distance'].median():,.0f}")
    lines.append("")

    lines.append("## Price by drivetrain")
    price_by_drive = df.groupby("Drive")["Price"].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
    lines.append(price_by_drive.to_markdown())
    lines.append("")

    lines.append("## Top 10 makes by volume & median price")
    make_stats = (
        df.groupby("Make")
        .agg(count=("Price", "size"), median_price=("Price", "median"), median_year=("Year", "median"))
        .sort_values("count", ascending=False)
        .head(10)
    )
    lines.append(make_stats.to_markdown())
    lines.append("")

    corr_cols = ["Price", "Year", "Engine", "Distance", "Cylinders", "Age", "Distance_per_year"]
    corr = df[corr_cols].corr(numeric_only=True)["Price"].drop("Price")
    lines.append("## Numeric correlations vs Price")
    lines.append(corr.to_frame("corr").to_markdown())

    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_feature_frames(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    categorical_cols = [
        "Make",
        "Model",
        "Style",
        "Fuel",
        "Color",
        "Transmission",
        "Drive",
        "Wheel",
        "Make_Model",
        "Make_Style",
        "Fuel_Transmission",
    ]

    train_cat = train_df.drop(columns=["ID"])
    test_cat = test_df.drop(columns=["ID"])

    # Preserve categorical dtype for CatBoost
    for col in categorical_cols:
        train_cat[col] = train_cat[col].astype(str)
        test_cat[col] = test_cat[col].astype(str)

    def make_freq_encoding(col: str) -> Dict[str, float]:
        freq = (
            pd.concat([train_cat[col], test_cat[col]])
            .value_counts(normalize=True)
            .to_dict()
        )
        return freq

    freq_maps = {col: make_freq_encoding(col) for col in categorical_cols}

    train_enc = train_cat.copy()
    test_enc = test_cat.copy()

    target_enc_cols = [
        "Make",
        "Model",
        "Style",
        "Fuel",
        "Color",
        "Make_Model",
        "Make_Style",
        "Fuel_Transmission",
    ]
    train_enc, test_enc = add_target_encodings(train_enc, test_enc, y, target_enc_cols)

    for col in categorical_cols:
        encoder = LabelEncoder()
        combined = pd.concat([train_cat[col], test_cat[col]], axis=0)
        encoder.fit(combined)
        train_enc[col] = encoder.transform(train_cat[col])
        test_enc[col] = encoder.transform(test_cat[col])

        train_enc[f"{col}_freq"] = train_cat[col].map(freq_maps[col])
        test_enc[f"{col}_freq"] = test_cat[col].map(freq_maps[col])

    return train_cat, test_cat, train_enc, test_enc, categorical_cols


def evaluate_models(
    X_enc: pd.DataFrame,
    y: pd.Series,
    X_cat: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {"lightgbm": [], "xgboost": [], "catboost": []}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # LightGBM + XGBoost use encoded frame
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_enc), 1):
        X_tr, X_val = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lgbm = LGBMRegressor(
            objective="mae",
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=80,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=fold * 17,
        )
        lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="mae")
        pred_lgbm = lgbm.predict(X_val)
        metrics["lightgbm"].append(mean_absolute_error(y_val, pred_lgbm))

        xgb = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            reg_alpha=0.2,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=fold * 19,
            n_jobs=-1,
            eval_metric="mae",
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        pred_xgb = xgb.predict(X_val)
        metrics["xgboost"].append(mean_absolute_error(y_val, pred_xgb))

    # CatBoost with categorical columns
    cat_features = categorical_cols
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cat), 1):
        X_tr, X_val = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        cat_model = CatBoostRegressor(
            loss_function="MAE",
            iterations=1800,
            depth=8,
            learning_rate=0.035,
            l2_leaf_reg=4,
            subsample=0.85,
            colsample_bylevel=0.6,
            random_seed=fold * 23,
            verbose=False,
        )
        cat_model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features),
            eval_set=Pool(X_val, y_val, cat_features=cat_features),
            use_best_model=True,
        )
        pred_cat = cat_model.predict(X_val)
        metrics["catboost"].append(mean_absolute_error(y_val, pred_cat))

    return {model: float(np.mean(scores)) for model, scores in metrics.items()}


def evaluate_holdout_models(
    X_enc: pd.DataFrame,
    y: pd.Series,
    X_cat: pd.DataFrame,
    categorical_cols: List[str],
    test_size: float = 0.2,
) -> Dict[str, float]:
    idx = np.arange(len(X_enc))
    train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=42)

    X_tr_enc, X_val_enc = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
    X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    results: Dict[str, float] = {}

    lgbm = LGBMRegressor(
        objective="mae",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=96,
        subsample=0.85,
        colsample_bytree=0.75,
        random_state=42,
    )
    lgbm.fit(X_tr_enc, y_tr)
    results["lightgbm"] = mean_absolute_error(y_val, lgbm.predict(X_val_enc))

    xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
    )
    xgb.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
    results["xgboost"] = mean_absolute_error(y_val, xgb.predict(X_val_enc))

    cat_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=2200,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=4,
        subsample=0.9,
        colsample_bylevel=0.7,
        random_seed=42,
        verbose=False,
    )
    cat_model.fit(
        Pool(X_tr_cat, y_tr, cat_features=categorical_cols),
        eval_set=Pool(X_val_cat, y_val, cat_features=categorical_cols),
        use_best_model=True,
    )
    results["catboost"] = mean_absolute_error(y_val, cat_model.predict(X_val_cat))

    return results


def fit_best_model(
    model_name: str,
    X_enc: pd.DataFrame,
    X_cat: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
) -> Tuple[np.ndarray, str]:
    if model_name == "catboost":
        model = CatBoostRegressor(
            loss_function="MAE",
            iterations=2200,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=4,
            subsample=0.9,
            colsample_bylevel=0.7,
            random_seed=42,
            verbose=False,
        )
        model.fit(Pool(X_cat, y, cat_features=categorical_cols))
        return model, "cat"

    if model_name == "lightgbm":
        model = LGBMRegressor(
            objective="mae",
            n_estimators=1800,
            learning_rate=0.025,
            num_leaves=96,
            subsample=0.85,
            colsample_bytree=0.75,
            random_state=42,
        )
        model.fit(X_enc, y)
        return model, "enc"

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_enc, y)
    return model, "enc"


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    train = _prepare_dataframe(train_raw, is_train=True)
    test = _prepare_dataframe(test_raw, is_train=False)

    generate_eda_report(train)

    y = train["Price"]
    train_feat = train.drop(columns=["Price"])
    train_feat, test = apply_shared_stats(train_feat, test)

    train_cat, test_cat, train_enc, test_enc, categorical_cols = build_feature_frames(train_feat, test, y)

    metrics_cv = evaluate_models(train_enc, y, train_cat, categorical_cols)
    holdout_metrics = evaluate_holdout_models(train_enc, y, train_cat, categorical_cols)
    metrics_payload = {"cv": metrics_cv, "holdout": holdout_metrics}
    (ARTIFACTS_DIR / "model_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print("Cross-validated MAE:", metrics_cv)
    print("Hold-out MAE:", holdout_metrics)

    best_model_name = min(holdout_metrics, key=holdout_metrics.get)
    print(f"Best model based on hold-out: {best_model_name}")

    best_model, variant = fit_best_model(best_model_name, train_enc, train_cat, y, categorical_cols)
    if variant == "cat":
        preds = best_model.predict(Pool(test_cat, cat_features=categorical_cols))
    else:
        preds = best_model.predict(test_enc)

    submit = pd.DataFrame({"ID": test_raw["ID"], "Predict": preds})
    before = len(submit)
    submit = submit.drop_duplicates(subset="ID", keep="first")
    if len(submit) != before:
        print(f"Dropped {before - len(submit)} duplicate IDs from submission.")
    submit.to_csv(ROOT / "submit.csv", index=False)
    print("Saved submit.csv with shape:", submit.shape)


if __name__ == "__main__":
    main()

