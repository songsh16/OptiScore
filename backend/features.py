"""
선수 시장 가치 예측을 위한 공용 피처 엔지니어링 모듈.

기존 `final_model.ipynb`는 학습 코드(배치, groupby-shift 기반)와
`predict_custom_player()`(단일 선수 입력, if-in-columns 방식) 두 곳에
같은 로직(포지션 그룹핑, 변화량 대상 스탯 목록 등)을 따로 손코딩해서
`stats_for_change`의 '태클' vs '태클 성공' 같은 불일치 버그가 생겼었다.
이 모듈이 그 로직의 단일 소스이며, `ml/notebooks/model_training.ipynb`
(학습)와 `backend/app.py`(서빙)가 모두 이 모듈을 가져다 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_5_LEAGUES = ["Bundesliga", "LaLiga", "Premier League", "Serie A", "Ligue 1"]

# FotMob/Transfermarkt 원본 컬럼명 -> 학습에 쓰는 한글 컬럼명.
# 'Successful Dribbles.1'은 엑셀 원본에 중복 헤더로 생긴 쓰레기 컬럼이라 제외한다.
RENAME_MAP = {
    "Present market value": "TM_시장가치_EUR",
    "Days Injured": "InjuryDays",
    "Name": "name_key",
    "Goals": "득점",
    "Shots": "슛",
    "Shots on Target": "유효 슈팅",
    "Assists": "어시스트",
    "Successful Passes": "성공한 패스",
    "Pass Accuracy": "패스 정확도",
    "Accurate Long Passes": "정확한 긴 패스",
    "Long Pass Accuracy": "롱 패스 정확도",
    "Chances Created": "기회 창출",
    "Successful Crosses": "성공한 크로스",
    "Cross Accuracy": "크로스 정확도",
    "Successful Dribbles": "드리블 성공",
    "Touches": "터치",
    "Touches in Opponent’s Box": "상대편 박스 내에서의 터치",
    "Possession Lost": "볼 뺏김",
    "Fouls Won": "획득한 파울",
    "Penalties Won": "패널티킥 받음",
    "Penalties Conceded": "패널티킥 허용",
    "Tackles": "태클",
    "Duels Won": "볼 경합 성공",
    "Duel Success %": "볼 경합 성공 %",
    "Aerial Duels Won": "공중 볼 경합 성공",
    "Aerial Duel Success %": "공중 볼 경합 성공 %",
    "Interceptions": "가로채기",
    "Blocked Shots": "막힌 슛",
    "Fouls Committed": "반칙",
    "Recoveries": "회복",
    "Possession in Attacking Third": "공격 지역 점유율",
    "Dribbled Past": "드리블로 제침",
    "Yellow Cards": "경고",
    "Red Cards": "퇴장",
    "Birth": "생년월일",
    "Position": "포지션",
    "Age at Season End": "시즌종료시점만나이",
    "League": "리그",
    "Season": "시즌",
}

# 시즌 간 변화량(전시즌 대비)을 피처로 쓰는 스탯들. 학습/서빙 어디서든 이 목록 하나만 본다.
STATS_FOR_CHANGE = ["득점", "어시스트", "기회 창출", "슛", "태클", "가로채기", "터치", "성공한 패스"]

# 사람이 이해할 수 있는 "원본 스탯" 입력 필드 (프런트엔드 예측 폼에 노출되는 목록).
# 메타 정보(포지션/나이/작년 시장가치/부상일수)는 별도 필드로 다룬다.
RAW_STAT_FIELDS = [
    "득점", "슛", "유효 슈팅", "어시스트", "성공한 패스", "패스 정확도",
    "정확한 긴 패스", "롱 패스 정확도", "기회 창출", "성공한 크로스", "크로스 정확도",
    "드리블 성공", "터치", "상대편 박스 내에서의 터치", "볼 뺏김", "획득한 파울",
    "패널티킥 받음", "패널티킥 허용", "태클", "볼 경합 성공", "볼 경합 성공 %",
    "공중 볼 경합 성공", "공중 볼 경합 성공 %", "가로채기", "막힌 슛", "반칙",
    "회복", "공격 지역 점유율", "드리블로 제침", "경고", "퇴장",
]

POSITION_GROUPS = [
    "공격형 미드필더", "수비형 미드필더", "스트라이커", "윙어",
    "중앙 미드필더", "중앙 수비수", "측면 미드필더", "측면 수비수",
]

_META_COLS = {"리그", "시즌", "name_key", "생년월일", "포지션"}


def clean_numeric(value):
    """'62%' -> 62.0, '-' -> 0.0, 숫자 문자열 -> float. 이미 숫자면 그대로."""
    if isinstance(value, str):
        if "%" in value:
            return float(value.replace("%", ""))
        if value.strip() == "-":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value


def group_position_detailed(position) -> str:
    """
    Transfermarkt 원본처럼 세부 포지션 문자열("오른쪽 윙어", "왼쪽 수비수" 등)이
    들어오면 8개 그룹 중 하나로 묶는다. 이미 그룹명 자체("측면 수비수" 등)가
    들어온 경우도 그대로 인정한다 — 예측 폼의 포지션 드롭다운이 그룹명을 직접
    보내는데, 원래 이 함수는 "왼쪽 수비수"/"오른쪽 수비수" 같은 세부 명칭만
    인식하고 "측면 수비수"라는 그룹명 자체는 매칭하지 못해 항상 '기타'로
    떨어지는 버그가 있었다.
    """
    raw = str(position).strip()
    if raw in POSITION_GROUPS:
        return raw

    pos = raw.lower()
    if "중앙 수비수" in pos:
        return "중앙 수비수"
    if "수비형 미드필더" in pos:
        return "수비형 미드필더"
    if "공격형 미드필더" in pos:
        return "공격형 미드필더"
    if "중앙 미드필더" in pos:
        return "중앙 미드필더"
    if "윙어" in pos:
        return "윙어"
    if "스트라이커" in pos or "중앙 공격수" in pos:
        return "스트라이커"
    if "왼쪽 미드필더" in pos or "오른쪽 미드필더" in pos:
        return "측면 미드필더"
    if "왼쪽 수비수" in pos or "오른쪽 수비수" in pos:
        return "측면 수비수"
    return "기타"


def _denoise_side_midfielder(df: pd.DataFrame) -> pd.DataFrame:
    """
    측면 미드필더 표본이 너무 작은 문제(다른 포지션 대비 R²가 크게 낮음)를
    완화하기 위한 시즌 라벨 정리. 어떤 선수의 측면 미드필더 시즌 수가 그 외
    시즌 수보다 적거나 같고, 그 외 시즌들이 전부 같은 한 그룹으로 일치하면
    (예: 6시즌 중 4시즌 윙어, 2시즌만 측면 미드필더) 소수/동률 쪽을 그
    그룹으로 재분류한다. 동률일 때도 측면 미드필더가 아닌 쪽으로 정리한다.

    그 외 시즌이 여러 그룹으로 나뉘어 있으면(예: 실제로 커리어 중반에 포지션이
    완전히 바뀐 선수) 어느 쪽이 "진짜"인지 판단할 근거가 없으므로 건드리지
    않는다.
    """
    df = df.copy()
    for _, idx in df.groupby("name_key").groups.items():
        sub = df.loc[idx]
        is_smf = sub["포지션_그룹"] == "측면 미드필더"
        smf_count = int(is_smf.sum())
        if smf_count == 0 or smf_count == len(sub):
            continue  # 측면 미드필더가 없거나, 전부 측면 미드필더면 그대로 둔다

        others = sub.loc[~is_smf, "포지션_그룹"]
        other_unique = others.unique()
        if len(other_unique) == 1 and smf_count <= len(others):
            df.loc[sub.index[is_smf], "포지션_그룹"] = other_unique[0]

    return df


def load_and_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    """원본 엑셀 -> 컬럼명 정리, 숫자 클리닝, 5대 리그/필드 플레이어로 필터링."""
    df = raw_df.drop(columns=["Successful Dribbles.1"], errors="ignore")
    df = df.rename(columns=RENAME_MAP)

    for col in df.columns:
        if col not in _META_COLS:
            df[col] = df[col].apply(clean_numeric)

    df = df[df["리그"].isin(TOP_5_LEAGUES)].copy()
    df["TM_시장가치_EUR"] = pd.to_numeric(df["TM_시장가치_EUR"], errors="coerce")
    df["InjuryDays"] = pd.to_numeric(df["InjuryDays"], errors="coerce").fillna(0)
    df.dropna(subset=["TM_시장가치_EUR", "포지션"], inplace=True)
    df = df[df["TM_시장가치_EUR"] > 0]
    df = df[df["포지션"] != "골키퍼"].copy()

    df["포지션_그룹"] = df["포지션"].apply(group_position_detailed)
    df = df[df["포지션_그룹"] != "기타"].copy()
    df = _denoise_side_midfielder(df)
    return df


def engineer_batch(df: pd.DataFrame) -> pd.DataFrame:
    """선수(name_key)별 시계열 순서를 이용한 배치 피처 엔지니어링 (학습용)."""
    df = df.sort_values(by=["name_key", "시즌"]).copy()

    df["나이_제곱"] = df["시즌종료시점만나이"] ** 2

    # 커리어 시즌차수: 이 선수가 데이터셋에서 몇 번째 시즌인지 (1부터 시작).
    # 나이와는 별개로 "얼마나 이 리그에서 검증됐는가"를 나타내는 신호.
    df["커리어_시즌차수"] = df.groupby("name_key").cumcount() + 1
    has_prev = df["커리어_시즌차수"] > 1

    for stat in STATS_FOR_CHANGE:
        df[f"전시즌_{stat}"] = df.groupby("name_key")[stat].shift(1).fillna(0)
        df[f"{stat}_변화량"] = df[stat] - df[f"전시즌_{stat}"]
        # 단일 시즌 변화량은 노이즈가 크므로, 최근 2개 시즌 평균도 함께 본다.
        # 직전 시즌 기록이 없는 데뷔 시즌은 이번 시즌 값 자체를 평균으로 쓴다.
        df[f"{stat}_2시즌평균"] = np.where(
            has_prev, (df[stat] + df[f"전시즌_{stat}"]) / 2, df[stat]
        )

    df["나이x득점_변화량"] = df["시즌종료시점만나이"] * df["득점_변화량"]
    df["슛_대비_득점율"] = np.divide(
        df["득점"], df["슛"], out=np.zeros_like(df["득점"], dtype=float), where=df["슛"] != 0
    )

    df["전시즌_시장가치_EUR"] = df.groupby("name_key")["TM_시장가치_EUR"].shift(1).fillna(0)
    df["log_전시즌_시장가치"] = np.log1p(df["전시즌_시장가치_EUR"])
    df["log_현재_시장가치"] = np.log1p(df["TM_시장가치_EUR"])
    df["시장가치_변화량"] = df["TM_시장가치_EUR"] - df["전시즌_시장가치_EUR"]

    # 리그 원-핫: 5대 리그라도 평균 몸값 수준(특히 프리미어리그의 "재정 프리미엄")이
    # 달라서, 같은 스탯이라도 리그에 따라 시장가치가 체계적으로 다르다.
    for league in TOP_5_LEAGUES:
        df[f"리그_{league}"] = (df["리그"] == league).astype(float)

    # 포지션 원-핫: 포지션별로 모델을 8개 따로 학습하는 대신, 포지션을 피처로
    # 넣은 단일 통합 모델 하나가 훨씬 낫다는 게 실험으로 확인됐다 (특히 표본이
    # 적은 측면 미드필더가 전체 11,637행에서 학습된 나이 곡선/리그 효과 등의
    # 혜택을 받아 GroupKFold CV R²가 44.3% -> 65.1%로 뛰었다). 자세한 비교는
    # ml/reports/model_evaluation.md 참고.
    for pos in POSITION_GROUPS:
        df[f"포지션_{pos}"] = (df["포지션_그룹"] == pos).astype(float)

    return df


# 학습 시 X에서 제외하는 컬럼 (타깃/식별자/중간 계산용 컬럼).
NON_FEATURE_COLS = [
    "리그", "TM_시장가치_EUR", "전시즌_시장가치_EUR", "시장가치_변화량", "log_현재_시장가치",
    "선수명", "생년월일", "name_key", "시즌", "포지션", "포지션_그룹",
] + [f"전시즌_{stat}" for stat in STATS_FOR_CHANGE]


def engineer_single(
    position: str,
    age: float,
    prev_market_value: float,
    injury_days: float,
    current_stats: dict,
    prev_stats: dict,
    feature_cols: list[str],
    league: str | None = None,
) -> pd.DataFrame:
    """
    실사용자가 입력한 원본 스탯 한 건을 학습 때와 동일한 피처로 변환한다.
    (노트북의 predict_custom_player()를 프로덕션 코드로 승격한 버전)

    커리어 시즌차수는 실제로는 알 수 없으므로, prev_stats가 하나라도 있으면
    "2년차 이상"(has_prev=True), 없으면 "데뷔 시즌"으로 근사한다.
    """
    current_stats = current_stats or {}
    prev_stats = prev_stats or {}
    has_prev = bool(prev_stats)

    row = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    for stat, value in current_stats.items():
        if stat in row.columns:
            row[stat] = value

    if "InjuryDays" in row.columns:
        row["InjuryDays"] = injury_days
    if "시즌종료시점만나이" in row.columns:
        row["시즌종료시점만나이"] = age
    if "나이_제곱" in row.columns:
        row["나이_제곱"] = age ** 2
    if "log_전시즌_시장가치" in row.columns:
        row["log_전시즌_시장가치"] = np.log1p(max(prev_market_value, 0))
    if "커리어_시즌차수" in row.columns:
        row["커리어_시즌차수"] = 2 if has_prev else 1

    for stat in STATS_FOR_CHANGE:
        change_col = f"{stat}_변화량"
        avg_col = f"{stat}_2시즌평균"
        curr_val = current_stats.get(stat, 0)
        prev_val = prev_stats.get(stat, 0)
        if change_col in row.columns:
            row[change_col] = curr_val - prev_val
        if avg_col in row.columns:
            row[avg_col] = (curr_val + prev_val) / 2 if has_prev else curr_val

    if "나이x득점_변화량" in row.columns:
        goal_change = current_stats.get("득점", 0) - prev_stats.get("득점", 0)
        row["나이x득점_변화량"] = age * goal_change

    if "슛_대비_득점율" in row.columns:
        goals = current_stats.get("득점", 0)
        shots = current_stats.get("슛", 0)
        row["슛_대비_득점율"] = goals / shots if shots > 0 else 0

    if league:
        league_col = f"리그_{league}"
        if league_col in row.columns:
            row[league_col] = 1.0

    position_col = f"포지션_{group_position_detailed(position)}"
    if position_col in row.columns:
        row[position_col] = 1.0

    return row
