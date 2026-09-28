# 데이터 수집 파이프라인

`data/player.xlsx`(모델 학습에 쓰인 최종 데이터셋)는 아래 3단계를 거쳐 만들어졌다.
이 폴더의 스크립트는 원래 6~10개의 노트북에 흩어져 있던 초안 코드를
리그별로 반복 실행 가능한 형태로 정리한 것이다 (원본 초안은 git 히스토리에는
남기지 않았고, 로직만 정리해서 옮겼다).

## 1단계 — FotMob 시즌 스탯 크롤링 (`scrape_fotmob.py`)
Selenium으로 FotMob 리그별 통계 페이지를 순회하며 선수별 90분당 스탯을 수집한다.
5대 리그(분데스리가/라리가/프리미어리그/세리에A/리그1) × 여러 시즌을 각각 실행해서
얻은 결과를 병합한 것이 `data/raw/fotmob_*.xlsx`다.

## 2단계 — Transfermarkt 선수 ID 매칭 (`match_transfermarkt_ids.py`)
FotMob 선수명으로 Transfermarkt 검색 API를 호출해 선수 ID를 찾는다.
결과: `data/raw/transfermarkt_player_ids_<league>.csv`

## 3단계 — 시장 가치 이력 수집 (`fetch_market_values.py`)
2단계에서 찾은 ID로 Transfermarkt의 시장 가치 변동 API를 호출해
시즌별 시장 가치(`Present market value`)를 가져온다.

## 최종 병합
1~3단계 결과와 FotMob 스탯을 선수명(+시즌) 기준으로 조인하고 수작업 보정을 거친
결과가 `data/player.xlsx`다. 조인/보정 스크립트 자체는 일회성 수작업이 많이 섞여
있어 재현 스크립트로 정리하지 않고, 최종 산출물만 `data/player.xlsx`로 보존했다.

## 참고
- 이 스크립트들은 외부 사이트에 실제 HTTP 요청을 보낸다. 포트폴리오 열람 목적으로
  코드 흐름만 보려면 실행하지 않아도 된다 — 실제 수집 결과는 `data/raw/`와
  `data/player.xlsx`에 이미 있다.
- Transfermarkt/FotMob 페이지 구조나 CSS 클래스명은 시점에 따라 바뀔 수 있으므로
  재실행 시 선택자를 다시 확인해야 할 수 있다.
