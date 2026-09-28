# 데이터 사전

## `player.xlsx` (학습 원본, 41개 컬럼)
5대 리그(분데스리가/라리가/프리미어리그/세리에A/리그1) 필드 플레이어의 시즌별
스탯 + Transfermarkt 시장 가치. 수집 과정은 [`../ml/data_pipeline/README.md`](../ml/data_pipeline/README.md)
참고.

| 컬럼 | 의미 |
|---|---|
| League, Season, Name | 리그, 시즌(예: `2016-2017`), 선수명 |
| Goals ~ Red Cards | FotMob 90분당 스탯 (득점, 슛, 패스, 드리블, 수비 지표 등) |
| Days Injured | 해당 시즌 부상 일수 |
| Present market value | 해당 시즌 시점 Transfermarkt 시장 가치 (EUR) |
| Birth, Position, Age at Season End | 생년월일, 포지션(세부), 시즌 종료 시점 만 나이 |

`backend/features.py`의 `RENAME_MAP`이 이 영문 컬럼을 학습에 쓰는 한글 컬럼명으로 바꾼다.
(`'Successful Dribbles.1'`은 엑셀 원본의 중복 헤더로 생긴 컬럼이라 로딩 시 제거한다.)

## `player_with_pred.xlsx`
위 데이터에 `Predicted market value` 컬럼을 추가한 결과물. `backend/app.py`가 이 파일을
읽어서 `/api/players`, `/api/player`에 응답한다. `python ml/train.py`를 실행하면
`backend/models/model_bundle.pkl`과 함께 다시 생성된다.

## `raw/`
`ml/data_pipeline/`의 원시 수집 결과 (FotMob 스탯 엑셀, Transfermarkt 선수 ID 매핑,
리그별 유니크 선수 목록). `player.xlsx`를 만드는 데 쓰인 중간 산출물이며, 재현성을
위해 보존한다.
