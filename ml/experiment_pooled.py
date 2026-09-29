"""
실험: 포지션 8개를 따로 학습하는 대신, 포지션을 피처로 넣은 단일 통합 모델
하나가 더 나은지 검증한다. 특히 표본이 적은 측면 미드필더(343건)가 전체
11,637행에서 배운 패턴(나이 곡선, 리그 효과 등)의 혜택을 받을 수 있는지가
핵심 질문.

기존 포지션별 모델과 똑같은 GroupKFold(4-fold) 방식으로 평가해서 공정하게
비교한다. 결과는 코드에 반영하지 않고 출력만 한다 — 실제로 나아야 train.py에
반영.

실행: python ml/experiment_pooled.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
import features  # noqa: E402
import train as train_module  # noqa: E402
from model_factory import MODEL_FACTORIES  # noqa: E402

N_FOLDS = 4

# 기존 포지션별 튜닝값의 중간값 정도로 잡은 합리적인 기본 파라미터 (공정 비교를
# 위한 1차 검증용 — 여기서 가능성이 보이면 그때 Optuna로 제대로 재탐색한다).
XGB_PARAMS = dict(
    n_estimators=800, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=1e-3, reg_lambda=1e-2,
)
LGBM_PARAMS = dict(
    n_estimators=800, max_depth=6, num_leaves=31, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
    reg_alpha=1e-3, reg_lambda=1e-2,
)


def build_pooled_xy(df: pd.DataFrame):
    df = df.copy()
    for pos in features.POSITION_GROUPS:
        df[f"포지션_{pos}"] = (df["포지션_그룹"] == pos).astype(float)
    y = df["log_현재_시장가치"]
    drop_cols = [c for c in features.NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols).fillna(0)
    return X, y, df["포지션_그룹"]


def cv_evaluate_pooled(X, y, groups, position_labels, model_type, params):
    gkf = GroupKFold(n_splits=N_FOLDS)
    per_position = {p: [] for p in features.POSITION_GROUPS}
    overall = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        pos_test = position_labels.iloc[test_idx]

        inner = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=0)
        fit_idx, val_idx = next(inner.split(X_train, y_train, groups_train))
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]

        model, kwargs = MODEL_FACTORIES[model_type](params, X_val, y_val)
        model.fit(X_fit, y_fit, **kwargs)
        pred = model.predict(X_test)

        overall.append(r2_score(y_test, pred))
        for pos in features.POSITION_GROUPS:
            mask = (pos_test == pos).values
            if mask.sum() >= 5:
                per_position[pos].append(r2_score(y_test[mask], pred[mask]))

    return overall, per_position


def main():
    print("데이터 로딩...")
    df = train_module.load_data()
    X, y, position_labels = build_pooled_xy(df)
    groups = df["name_key"]
    print(f"통합 데이터: {len(df)}행, {groups.nunique()}명, 피처 {X.shape[1]}개\n")

    results = {}
    for model_type, params in [("xgboost", XGB_PARAMS), ("lightgbm", LGBM_PARAMS)]:
        print(f"=== 통합 모델 ({model_type}) ===")
        overall, per_position = cv_evaluate_pooled(X, y, groups, position_labels, model_type, params)
        print(f"전체 풀링 R² (참고용, 포지션 섞임): {np.mean(overall):.2%}")
        results[model_type] = {pos: float(np.mean(scores)) for pos, scores in per_position.items()}
        for pos, scores in per_position.items():
            print(f"  {pos}: {np.mean(scores):.2%} (±{np.std(scores):.2%})")
        print()

    # 기존 포지션별 전용 모델과 비교
    current = json.load(open(ROOT / "ml" / "reports" / "results.json", encoding="utf-8"))
    print("=== 비교: 기존 포지션별 전용 모델 vs 통합 모델 ===")
    print(f"{'포지션':<10} {'전용모델 CV R²':>14} {'통합(xgb)':>12} {'통합(lgbm)':>12} {'승자':>10}")
    pooled_wins = 0
    for pos in features.POSITION_GROUPS:
        specialized = current[pos]["cv_split"]["r2_mean"]
        pooled_xgb = results["xgboost"][pos]
        pooled_lgbm = results["lightgbm"][pos]
        best_pooled = max(pooled_xgb, pooled_lgbm)
        winner = "통합" if best_pooled > specialized else "전용"
        if winner == "통합":
            pooled_wins += 1
        print(f"{pos:<10} {specialized:>13.1%} {pooled_xgb:>11.1%} {pooled_lgbm:>11.1%} {winner:>10}")
    print(f"\n통합 모델이 이긴 포지션: {pooled_wins}/8")

    with open(ROOT / "ml" / "reports" / "pooled_experiment.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
