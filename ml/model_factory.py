"""XGBoost/LightGBM 모델 생성 헬퍼. ml/train.py와 ml/tune.py가 공용으로 쓴다."""

from __future__ import annotations

import lightgbm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def make_xgb(params: dict, X_val, y_val):
    model = XGBRegressor(
        objective="reg:squarederror", **params, random_state=42, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="rmse", verbosity=0,
    )
    return model, {"eval_set": [(X_val, y_val)], "verbose": False}


def make_lgbm(params: dict, X_val, y_val):
    model = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbosity=-1)
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "callbacks": [lightgbm.early_stopping(50, verbose=False)],
    }
    return model, fit_kwargs


MODEL_FACTORIES = {"xgboost": make_xgb, "lightgbm": make_lgbm}
