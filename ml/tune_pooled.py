"""
통합(pooled) 모델 하이퍼파라미터 재탐색. ml/experiment_pooled.py에서 기본
파라미터로도 8개 포지션 전부 개선되는 걸 확인한 뒤, 제대로 된 파라미터로
Optuna 재탐색을 돌린다.

목적함수는 "포지션별 CV R² 평균"(macro-average) — 표본이 많은 포지션에
치우치지 않고 8개 포지션 전체가 고르게 잘 되는 파라미터를 찾기 위함.

실행: python ml/tune_pooled.py --trials 40
출력: ml/reports/best_params_pooled.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
import features  # noqa: E402
import train as train_module  # noqa: E402
from model_factory import make_lgbm, make_xgb  # noqa: E402
from experiment_pooled import build_pooled_xy  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_FOLDS = 4


def macro_cv_score(X, y, groups, position_labels, model_ctor) -> float:
    gkf = GroupKFold(n_splits=N_FOLDS)
    per_position = {p: [] for p in features.POSITION_GROUPS}

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        pos_test = position_labels.iloc[test_idx]

        inner = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=0)
        fit_idx, val_idx = next(inner.split(X_train, y_train, groups_train))
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]

        model, kwargs = model_ctor(X_val, y_val)
        model.fit(X_fit, y_fit, **kwargs)
        pred = model.predict(X_test)

        for pos in features.POSITION_GROUPS:
            mask = (pos_test == pos).values
            if mask.sum() >= 5:
                per_position[pos].append(r2_score(y_test[mask], pred[mask]))

    position_means = [np.mean(v) for v in per_position.values() if v]
    return float(np.mean(position_means))


def suggest_xgb_params(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def suggest_lgbm_params(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        num_leaves=trial.suggest_int("num_leaves", 8, 200),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def main(n_trials: int):
    print("데이터 로딩...")
    df = train_module.load_data()
    X, y, position_labels = build_pooled_xy(df)
    groups = df["name_key"]
    print(f"통합 데이터: {len(df)}행, 피처 {X.shape[1]}개\n")

    def xgb_objective(trial):
        params = suggest_xgb_params(trial)
        return macro_cv_score(X, y, groups, position_labels, lambda xv, yv: make_xgb(params, xv, yv))

    def lgbm_objective(trial):
        params = suggest_lgbm_params(trial)
        return macro_cv_score(X, y, groups, position_labels, lambda xv, yv: make_lgbm(params, xv, yv))

    print(f"XGBoost 탐색 중 ({n_trials} trials)...")
    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=n_trials)
    print(f"  best macro R² = {xgb_study.best_value:.2%}")

    print(f"LightGBM 탐색 중 ({n_trials} trials)...")
    lgbm_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    lgbm_study.optimize(lgbm_objective, n_trials=n_trials)
    print(f"  best macro R² = {lgbm_study.best_value:.2%}")

    result = {
        "xgboost": {"macro_r2": xgb_study.best_value, "params": xgb_study.best_params},
        "lightgbm": {"macro_r2": lgbm_study.best_value, "params": lgbm_study.best_params},
    }
    out_path = ROOT / "ml" / "reports" / "best_params_pooled.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()
    main(args.trials)
