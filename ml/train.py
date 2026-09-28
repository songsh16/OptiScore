"""
포지션별 선수 시장 가치 예측 모델 학습.

기존 `final_model.ipynb`에서 고쳐진 것:
  1. 8개 포지션 모델을 전부 `backend/models/model_bundle.pkl`에 저장한다
     (기존 코드는 반복문 마지막 포지션 모델 하나만 저장하는 버그가 있었음).
  2. StandardScaler를 제거했다 (XGBoost 트리 분기는 피처별 임계값 비교라
     스케일링이 필요 없고, 기존 코드는 스케일러를 저장하지 않아 서빙 시
     학습 때와 다른 스케일의 입력이 들어가는 버그가 있었음).
  3. train_test_split을 선수(name_key) 단위로 묶어서 나눈다
     (기존 코드는 완전 랜덤 분할이라 같은 선수의 여러 시즌이 train/test에
     걸쳐 섞이는 데이터 누수가 있었음). 비교를 위해 "기존 방식(누수)" 결과도
     함께 계산해서 리포트에 남긴다.
  4. n_estimators가 최대 1900, max_depth가 최대 19인 기존 튜닝값을 그대로
     쓰되 early_stopping_rounds로 실제 트리 수를 제한해 과적합을 억제한다.

실행:
    python ml/train.py
출력:
    backend/models/model_bundle.pkl   ({포지션: {model, feature_cols}})
    data/player_with_pred.xlsx        (전체 선수에 대한 예측값 포함)
    ml/reports/results.json           (아래 model_evaluation.md 작성에 쓰는 원자료)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import features  # noqa: E402

DATA_PATH = ROOT / "data" / "player.xlsx"
BUNDLE_PATH = ROOT / "backend" / "models" / "model_bundle.pkl"
PRED_PATH = ROOT / "data" / "player_with_pred.xlsx"
RESULTS_PATH = ROOT / "ml" / "reports" / "results.json"

# 기존 노트북에서 가져온 포지션별 최적 하이퍼파라미터 (변경 없음).
BEST_PARAMS = {
    "공격형 미드필더": {"n_estimators": 1900, "learning_rate": 0.020762320977301275, "max_depth": 13, "subsample": 0.3856052273527386, "colsample_bytree": 0.7980135471467072, "reg_alpha": 0.00016282406660088258, "reg_lambda": 0.03482118513318421},
    "수비형 미드필더": {"n_estimators": 1400, "learning_rate": 0.022491096898892622, "max_depth": 19, "subsample": 0.427399045217705, "colsample_bytree": 0.7536284027955273, "reg_alpha": 6.530905564075279e-07, "reg_lambda": 0.028399498223413196},
    "스트라이커": {"n_estimators": 1600, "learning_rate": 0.011723795540417655, "max_depth": 15, "subsample": 0.4014413443070366, "colsample_bytree": 0.7460919009335952, "reg_alpha": 1.7387604754886692e-07, "reg_lambda": 3.6656893204449398e-06},
    "윙어": {"n_estimators": 1400, "learning_rate": 0.04132166762817925, "max_depth": 9, "subsample": 0.6216447365354127, "colsample_bytree": 0.5306921156821315, "reg_alpha": 6.592195514427946e-05, "reg_lambda": 8.811664062461144e-06},
    "중앙 미드필더": {"n_estimators": 1600, "learning_rate": 0.084860060235904, "max_depth": 5, "subsample": 0.7869147744468107, "colsample_bytree": 0.6814027657309146, "reg_alpha": 1.3579137923778576e-07, "reg_lambda": 3.829970428360115e-07},
    "중앙 수비수": {"n_estimators": 1100, "learning_rate": 0.020771872880752665, "max_depth": 4, "subsample": 0.5562494519644748, "colsample_bytree": 0.398997982165777, "reg_alpha": 0.061343450214043506, "reg_lambda": 0.46185038844108967},
    "측면 미드필더": {"n_estimators": 1700, "learning_rate": 0.06561899817995089, "max_depth": 3, "subsample": 0.675130546991308, "colsample_bytree": 0.8366998204911843, "reg_alpha": 3.871586781469543e-05, "reg_lambda": 1.5542377744930896e-07},
    "측면 수비수": {"n_estimators": 1400, "learning_rate": 0.023703783882164025, "max_depth": 6, "subsample": 0.9301629782976328, "colsample_bytree": 0.5744368741577944, "reg_alpha": 2.0849287673580608e-05, "reg_lambda": 0.0007417703250231236},
}

MIN_GROUP_SIZE = 20


def load_raw() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH)


def load_data() -> pd.DataFrame:
    raw = load_raw()
    df = features.load_and_clean(raw)
    df = features.engineer_batch(df)
    return df


def _make_xy(group_df: pd.DataFrame):
    y = group_df["log_현재_시장가치"]
    drop_cols = [c for c in features.NON_FEATURE_COLS if c in group_df.columns]
    X = group_df.drop(columns=drop_cols).fillna(0)
    return X, y


def _eval_naive_split(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    """기존 코드와 동일한 완전 랜덤 분할 (비교용, 데이터 누수 있음)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective="reg:squarederror", **params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, verbose=False)
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    return {
        "r2": r2_score(y_test, model.predict(X_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _eval_grouped_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict):
    """선수 단위 그룹 분할 (수정된 방식) + 조기 종료로 과적합 억제."""
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    # 조기 종료용 검증 세트도 선수 단위로 분리
    inner_splitter = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=42)
    fit_idx, val_idx = next(inner_splitter.split(X_train, y_train, groups_train))
    X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
    y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]

    model = XGBRegressor(
        objective="reg:squarederror", **params, random_state=42, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="rmse",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    log_pred_test = model.predict(X_test)
    y_pred = np.expm1(log_pred_test)
    y_true = np.expm1(y_test)

    metrics = {
        "r2": r2_score(y_test, log_pred_test),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "best_iteration": int(model.best_iteration) if model.best_iteration is not None else params["n_estimators"],
    }

    importances = sorted(zip(model.feature_importances_, X.columns), reverse=True)
    imp_df = pd.DataFrame(importances, columns=["importance", "feature"]).head(15)

    err_df = X_test.copy()
    err_df["실제_가치"] = y_true.values
    err_df["예측_가치"] = y_pred
    err_df["오차"] = err_df["예측_가치"] - err_df["실제_가치"]

    return model, metrics, imp_df, err_df


def train_all(df: pd.DataFrame) -> dict:
    results = {}
    bundle = {}

    for position in features.POSITION_GROUPS:
        group_df = df[df["포지션_그룹"] == position].copy()
        if len(group_df) < MIN_GROUP_SIZE:
            print(f"건너뜀: {position} (데이터 {len(group_df)}건)")
            continue

        params = BEST_PARAMS[position]
        X, y = _make_xy(group_df)
        groups = group_df["name_key"]

        naive_metrics = _eval_naive_split(X, y, params)
        model, grouped_metrics, imp_df, err_df = _eval_grouped_split(X, y, groups, params)

        bundle[position] = {"model": model, "feature_cols": list(X.columns)}
        results[position] = {
            "n_rows": len(group_df),
            "n_players": group_df["name_key"].nunique(),
            "naive_split": naive_metrics,
            "grouped_split": grouped_metrics,
            "top_features": imp_df.to_dict("records"),
            "overrated_top5": err_df.sort_values("오차", ascending=False)
                .assign(선수명=df.loc[err_df.index, "name_key"])[["선수명", "실제_가치", "예측_가치", "오차"]]
                .head(5).to_dict("records"),
            "underrated_top5": err_df.sort_values("오차", ascending=True)
                .assign(선수명=df.loc[err_df.index, "name_key"])[["선수명", "실제_가치", "예측_가치", "오차"]]
                .head(5).to_dict("records"),
        }
        print(
            f"{position}: 랜덤분할 R²={naive_metrics['r2']:.2%} -> 그룹분할 R²={grouped_metrics['r2']:.2%} "
            f"(n={len(group_df)}, best_iter={grouped_metrics['best_iteration']})"
        )

    return bundle, results


def predict_all(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    preds = np.full(len(df), np.nan)
    for position, info in bundle.items():
        mask = (df["포지션_그룹"] == position).values
        if not mask.any():
            continue
        X = df.loc[mask, info["feature_cols"]].fillna(0)
        preds[mask] = np.expm1(info["model"].predict(X))
    df = df.copy()
    df["Predicted market value"] = preds
    return df


def build_prediction_excel(raw_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    예측값을 원본(영문 컬럼) 데이터프레임에 병합한다. `backend/app.py`가 읽는
    `data/player_with_pred.xlsx`는 Birth/Position/Present market value 같은
    원본 컬럼명을 그대로 기대하므로, 학습용 한글 피처 컬럼이 섞인 내부
    엔지니어링 데이터프레임을 그대로 내보내면 안 된다.
    """
    predictions = pred_df[["name_key", "시즌", "Predicted market value"]].rename(
        columns={"name_key": "Name", "시즌": "Season"}
    )
    # data/player.xlsx에 예전 파이프라인이 남긴 빈 "Predicted market value" 컬럼이
    # 이미 있을 수 있으므로, 병합 전에 지워서 _x/_y 중복이 생기지 않게 한다.
    raw_df = raw_df.drop(columns=["Predicted market value"], errors="ignore")
    return raw_df.merge(predictions, on=["Name", "Season"], how="left")


def main():
    print("데이터 로딩/전처리...")
    raw = load_raw()
    df = features.engineer_batch(features.load_and_clean(raw))
    print(f"학습 대상: {len(df)}건, {df['name_key'].nunique()}명\n")

    bundle, results = train_all(df)

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"\n모델 번들 저장: {BUNDLE_PATH} ({len(bundle)}개 포지션)")

    pred_df = predict_all(df, bundle)
    out = build_prediction_excel(raw, pred_df)
    out.to_excel(PRED_PATH, index=False)
    print(f"예측 결과 저장: {PRED_PATH}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"평가 결과 저장: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
