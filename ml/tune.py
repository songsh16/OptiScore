"""
포지션별 하이퍼파라미터 재탐색 (Optuna, 선수 단위 GroupKFold 기준).

기존 `best_params_per_position`(ml/train.py)은 원래 프로젝트에서 랜덤 분할
기준으로 튜닝된 것으로 보인다 — max_depth가 최대 19까지 올라가는 등, 데이터
누수가 있는 지표를 최적화하다 보니 과도하게 복잡한 모델을 "최적"이라 판단했을
가능성이 크다. 이 스크립트는 선수 단위로 묶은 GroupKFold 교차검증 점수를
목적함수로 삼아, 실제로 일반화되는 하이퍼파라미터를 다시 찾는다.
탐색 공간도 max_depth<=8로 제한해 애초에 극단적인 트리가 나오지 않게 했다.

실행:
    python ml/tune.py --trials 40
출력:
    ml/reports/best_params.json  (포지션별 XGBoost/LightGBM 최적 파라미터 + CV 점수)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
import features  # noqa: E402
import train as train_module  # noqa: E402
from model_factory import make_lgbm, make_xgb  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_FOLDS = 4
MIN_GROUP_SIZE = 20


def cv_r2(model_and_fit_kwargs_ctor, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    """GroupKFold 교차검증 평균 R² (fold마다 내부 검증셋으로 조기 종료)."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        inner = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=0)
        fit_idx, val_idx = next(inner.split(X_train, y_train, groups_train))
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]

        model, fit_kwargs = model_and_fit_kwargs_ctor(X_val, y_val)
        model.fit(X_fit, y_fit, **fit_kwargs)
        scores.append(model.score(X_test, y_test))

    return float(np.mean(scores))


def suggest_xgb_params(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1000),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def suggest_lgbm_params(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1000),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        num_leaves=trial.suggest_int("num_leaves", 8, 128),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def tune_position(X, y, groups, n_trials: int) -> dict:
    def xgb_objective(trial):
        params = suggest_xgb_params(trial)
        return cv_r2(lambda xv, yv: make_xgb(params, xv, yv), X, y, groups)

    def lgbm_objective(trial):
        params = suggest_lgbm_params(trial)
        return cv_r2(lambda xv, yv: make_lgbm(params, xv, yv), X, y, groups)

    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)

    lgbm_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    lgbm_study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "xgboost": {"cv_r2": xgb_study.best_value, "params": xgb_study.best_params},
        "lightgbm": {"cv_r2": lgbm_study.best_value, "params": lgbm_study.best_params},
    }


def main(n_trials: int):
    print("데이터 로딩/전처리...")
    df = train_module.load_data()

    results = {}
    for position in features.POSITION_GROUPS:
        group_df = df[df["포지션_그룹"] == position].copy()
        if len(group_df) < MIN_GROUP_SIZE:
            continue

        X, y = train_module._make_xy(group_df)
        groups = group_df["name_key"]

        print(f"\n=== {position} (n={len(group_df)}) 튜닝 중 ({n_trials} trials x 2 models)... ===")
        result = tune_position(X, y, groups, n_trials)
        results[position] = result
        print(
            f"  XGBoost CV R²={result['xgboost']['cv_r2']:.2%} | "
            f"LightGBM CV R²={result['lightgbm']['cv_r2']:.2%}"
        )

    out_path = ROOT / "ml" / "reports" / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()
    main(args.trials)
