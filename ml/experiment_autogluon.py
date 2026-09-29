"""
AutoGluon 실험: XGBoost/LightGBM 외에 CatBoost·선형모델·신경망까지 자동으로
시도하고 스태킹 앙상블까지 구성해주는 AutoML이 통합(pooled) 모델보다
나은지 확인한다.

격리된 venv(ml/.autogluon_venv)에서 실행해야 한다 — AutoGluon은 numpy/pandas를
이 프로젝트의 나머지 부분보다 낮은 버전으로 요구해서, 전역 환경에 설치하면
numpy 2.2.6 -> 1.26.4 등으로 다운그레이드된다.

시간 예산이 커서(AutoGluon 자체가 여러 모델 + 스태킹을 시도) 전체 4-fold
GroupKFold 대신, 기존 실험들과 동일한 단일 GroupShuffleSplit(20% 테스트)으로
빠르게 비교한다 (참고용 — 완전히 공정한 비교는 아니지만 방향성은 확인 가능).

실행:
    ml/.autogluon_venv/Scripts/python.exe ml/experiment_autogluon.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
import features  # noqa: E402
import train as train_module  # noqa: E402
from experiment_pooled import build_pooled_xy  # noqa: E402

TIME_LIMIT_SEC = 300  # AutoGluon 학습 시간 예산


def main():
    from autogluon.tabular import TabularPredictor

    print("데이터 로딩...")
    df = train_module.load_data()
    X, y, position_labels = build_pooled_xy(df)
    groups = df["name_key"]
    print(f"통합 데이터: {len(df)}행, 피처 {X.shape[1]}개")

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    train_df = X.iloc[train_idx].copy()
    train_df["target"] = y.iloc[train_idx].values
    test_df = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx]
    pos_test = position_labels.iloc[test_idx]

    print(f"학습 {len(train_df)}행 / 테스트 {len(test_df)}행, 시간 예산 {TIME_LIMIT_SEC}초")
    t0 = time.time()
    predictor = TabularPredictor(label="target", problem_type="regression", eval_metric="r2", verbosity=2)
    # num_bag_folds=0, num_stack_levels=0: 배깅/스태킹에 쓰이는 Ray 분산 처리를
    # 비활성화한다. 이 환경(Windows 샌드박스)에서 Ray worker가
    # "access violation"으로 반복 크래시하는 걸 실제로 확인했다 — 대신 각
    # 모델(LightGBM/XGBoost/CatBoost/신경망 등)을 단일 홀드아웃 검증으로 학습하고
    # WeightedEnsemble로 묶는다.
    predictor.fit(
        train_df,
        time_limit=TIME_LIMIT_SEC,
        presets="good_quality",
        num_bag_folds=0,
        num_stack_levels=0,
        ag_args_fit={"num_cpus": 4},
    )
    print(f"학습 완료: {time.time()-t0:.0f}초")

    pred = predictor.predict(test_df).values
    overall_r2 = r2_score(y_test, pred)
    print(f"\n전체 풀링 R² (참고용): {overall_r2:.2%}")

    print("\n리더보드 (시도한 모델들):")
    print(predictor.leaderboard(silent=True).to_string())

    print("\n포지션별 R²:")
    per_position = {}
    for pos in features.POSITION_GROUPS:
        mask = (pos_test == pos).values
        if mask.sum() >= 5:
            r2 = r2_score(y_test[mask], pred[mask])
            per_position[pos] = float(r2)
            print(f"  {pos}: {r2:.2%} (n={mask.sum()})")

    import json
    with open(ROOT / "ml" / "reports" / "autogluon_experiment.json", "w", encoding="utf-8") as f:
        json.dump({"overall_r2": float(overall_r2), "per_position": per_position}, f, ensure_ascii=False, indent=2)
    print("\n저장 완료: ml/reports/autogluon_experiment.json")


if __name__ == "__main__":
    main()
