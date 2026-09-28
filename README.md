# OptiScore

5대 리그(분데스리가·라리가·프리미어리그·세리에A·리그1) 축구 선수의 시즌 스탯으로
Transfermarkt 시장 가치를 예측하는 풀스택 웹앱. 포지션 그룹(8개)마다 별도의
XGBoost 모델을 학습해서, 검색한 선수의 예상 시장 가치를 보여주고 임의의 스탯을
입력해 "이런 활약을 하면 몸값이 얼마나 될까"를 시뮬레이션해볼 수 있다.

## 데모 흐름
1. 선수 이름 검색 → 기본 정보(포지션/나이/현재 시장가치/모델 예측값) 확인
2. 원하는 선수를 즐겨찾기에 저장 (로그인 필요)
3. Prediction 페이지에서 포지션과 이번 시즌 스탯(+ 지난 시즌 스탯)을 입력해
   what-if 시장 가치 예측

## 아키텍처
```
frontend/  (바닐라 HTML/CSS/JS)  ──HTTP──▶  backend/app.py (Flask)
                                              │
                                              ├─ backend/features.py   (피처 엔지니어링, 학습·서빙 공용)
                                              ├─ backend/models/model_bundle.pkl  (포지션별 XGBoost 8개)
                                              └─ data/player_with_pred.xlsx (선수 조회용)

ml/train.py + ml/notebooks/model_training.ipynb  → 모델 학습, backend/models/model_bundle.pkl 생성
ml/data_pipeline/                                 → 원본 데이터 수집 스크립트 (FotMob/Transfermarkt)
```

## 실행 방법
```bash
pip install -r requirements.txt

# (모델을 이미 backend/models/model_bundle.pkl로 갖고 있다면 이 단계는 생략 가능)
python ml/train.py

cd backend
python app.py
# http://127.0.0.1:5000
```

## 이 프로젝트에서 고친 것들
원래 있던 버전(개인 캡스톤 프로젝트 초안)을 정리하면서 실제로 고친 버그·구조적
문제들. 자세한 재검증 수치는 [`ml/reports/model_evaluation.md`](ml/reports/model_evaluation.md) 참고.

1. **포지션별 모델이 실제로 안 쓰이던 버그**: 8개 포지션 모델을 각각 학습했지만
   `joblib.dump`로 저장되는 건 학습 루프의 마지막 포지션(측면 수비수) 모델
   하나뿐이었다 — 서비스는 스트라이커든 미드필더든 전부 그 모델 하나로 예측하고
   있었다. 지금은 `{포지션: 모델}` 딕셔너리 전체를 저장하고, `/api/predict`가
   입력된 포지션에 맞는 모델을 선택한다.
2. **스케일러 누락 버그**: 학습은 `StandardScaler`로 표준화한 데이터로 했지만
   저장된 번들엔 스케일러가 빠져 있어, 서빙 시 원본 스케일 입력이 학습 때와 다른
   기준으로 트리 분기와 비교되는 문제가 있었다. XGBoost는 트리 분기가 피처별
   임계값 비교라 스케일링이 애초에 불필요하므로, 스케일링 자체를 제거했다.
3. **데이터 누수**: `train_test_split`을 선수 단위 그룹 없이 완전 랜덤으로 나눠서
   같은 선수의 여러 시즌이 train/test에 걸쳐 섞였다. `GroupShuffleSplit`으로
   선수 단위 분할로 바꿔 재검증한 결과, 8개 중 6개 포지션에서 기존 R²가 실제보다
   낙관적으로 부풀려져 있었다 (예: 윙어 75.2% → 62.1%).
4. **과적합 위험**: 기존 하이퍼파라미터는 `max_depth` 최대 19, `n_estimators`
   최대 1,900으로 수천 건 규모 데이터엔 과도했다. `early_stopping_rounds`를
   적용해 실제 사용된 트리 수를 10~30% 수준으로 자동 제한했다.
5. **실사용 불가능한 예측 폼**: 프런트엔드가 내부 엔지니어링 피처명
   (`log_전시즌_시장가치`, `나이_제곱`, `득점_변화량` ...)을 그대로 입력 폼으로
   노출해서, 실사용자가 계산도 할 수 없는 값을 입력해야 했다. 지금은 원본 스탯
   (득점, 어시스트, 터치 ...)만 입력받고, 서버(`backend/features.py`)가 내부에서
   나머지 피처를 계산한다.
6. **피처 엔지니어링 중복 버그**: 학습 코드와 예측 함수가 각각 손코딩되어 있어서
   `'태클'` vs `'태클 성공'` 같은 오타로 커스텀 예측 시 특정 피처가 항상 0으로
   무시되는 버그가 있었다. 이제 `backend/features.py` 하나가 학습(`ml/train.py`)과
   서빙(`backend/app.py`) 양쪽의 단일 소스다.
7. **하드코딩된 API 주소**: 프런트엔드 곳곳에 `http://127.0.0.1:5000`이
   하드코딩되어 있어 배포 환경이 바뀌면 깨지는 구조였다. 같은 origin에서 서빙하는
   것을 전제로 상대 경로로 통일했다.
8. **폴더 구조**: 크롤링 초안 3벌(`chrolling/`, `code/`, `code2/`), 이중 중첩된
   앱 폴더, 479MB `.venv`, 161MB 백업 zip이 한 디렉터리에 섞여 있던 걸
   `backend/ frontend/ data/ ml/ docs/`로 정리했다. 초안들은 로직만 정리해서
   [`ml/data_pipeline/`](ml/data_pipeline/)로 옮기고 원본은
   [`docs/archive/_deprecated/`](docs/archive/_deprecated/)에 보관했다.

## 알려진 한계
- **측면 미드필더** 포지션은 표본이 356건(162명)으로 적어 R²가 40% 수준으로 낮다.
- 시즌 스탯만으로는 브랜드 가치·이적시장 하이프를 설명하지 못해, 슈퍼스타
  (메시, 음바페, 손흥민 등)는 일관되게 과소평가된다.
- 지금의 "선수 단위 그룹 분할"도 실제 배포 시나리오(과거 시즌으로 학습해 미래
  시즌을 예측)보다는 관대한 검증이다. 더 엄격하게 보려면 시즌 기준 시간 분할이
  필요하다.
- 프런트엔드 6개 HTML 페이지에 header/footer/button CSS가 상당 부분 중복되어
  있다. 기능상 문제는 없지만 `frontend/static/css`로 추출해 공용화하면 더
  깔끔해질 부분으로 남겨뒀다 (브라우저로 직접 확인하며 진행하는 게 안전해서
  이번 작업 범위에서는 보류).

## 기술 스택
- **Backend**: Flask, SQLite(회원/즐겨찾기), XGBoost, scikit-learn, pandas
- **Frontend**: 바닐라 HTML/CSS/JS (프레임워크 없음)
- **ML**: 포지션별 XGBoost 회귀, `GroupShuffleSplit` + early stopping

## 폴더 구조
```
backend/    Flask 앱, 피처 엔지니어링, 학습된 모델
frontend/   정적 HTML/CSS/JS
data/       학습 데이터셋 + 원시 수집 데이터
ml/         학습 스크립트/노트북, 데이터 수집 파이프라인, 평가 리포트
docs/       발표자료·진행노트 아카이브, 정리 과정에서 대체된 초안 보관
```
