import os
import sqlite3
from datetime import datetime, timedelta

import bcrypt
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS

import features

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
PLAYER_EXCEL = os.path.join(ROOT_DIR, "data", "player_with_pred.xlsx")
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "models", "model_bundle.pkl")

app = Flask(__name__)
app.secret_key = os.environ.get("OPTISCORE_SECRET_KEY", "dev-only-secret-key-change-me")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

# (프런트엔드와 같은 origin에서 서빙되므로 CORS는 사실 필요 없지만 기본만 켜둠)
CORS(app, supports_credentials=True)

# ------------------------------------------------------------------
# 0. 전역 모델 변수 — 포지션 그룹별로 하나씩 갖고 있는다.
#    (기존 버전은 학습 루프의 마지막 포지션 모델 하나만 저장/로딩해서
#     모든 포지션의 예측을 그 모델 하나로 처리하는 버그가 있었다.)
# ------------------------------------------------------------------
MODELS_BY_POSITION: dict[str, dict] = {}


def load_models():
    global MODELS_BY_POSITION
    if not os.path.exists(MODEL_BUNDLE_PATH):
        print(f"모델 번들을 찾지 못했습니다: {MODEL_BUNDLE_PATH}")
        MODELS_BY_POSITION = {}
        return

    try:
        MODELS_BY_POSITION = joblib.load(MODEL_BUNDLE_PATH)
        print(f"모델 로딩 성공: {len(MODELS_BY_POSITION)}개 포지션 ({', '.join(MODELS_BY_POSITION)})")
    except Exception as e:
        print("모델 로딩 실패:", e)
        MODELS_BY_POSITION = {}


# ------------------------------------------------------------------
# 1. DB 유틸
# ------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "users.db"))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    db.commit()
    db.close()


with app.app_context():
    init_db()
    load_models()


# ------------------------------------------------------------------
# 2. player_with_pred.xlsx 연동 유틸
# ------------------------------------------------------------------
players_df = None


def load_players_df():
    global players_df
    if players_df is not None:
        return
    df = pd.read_excel(PLAYER_EXCEL)
    df.columns = [str(c).strip().lower() for c in df.columns]
    players_df = df


def get_name_col(df):
    for col in ["name", "player_name", "player", "선수명"]:
        if col in df.columns:
            return col
    return None


def safe_str(v):
    if pd.isna(v):
        return ""
    return str(v)


# ------------------------------------------------------------------
# 3. 회원가입 / 로그인 / 세션
# ------------------------------------------------------------------
@app.post("/api/join")
def api_join():
    data = request.json or {}
    user_id = data.get("id")
    name = data.get("name")
    email = data.get("email")
    pw = data.get("password")

    if not user_id or not name or not email or not pw:
        return jsonify(ok=False, message="모든 필드를 입력해주세요."), 400

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    if cur.fetchone():
        db.close()
        return jsonify(ok=False, message="이미 존재하는 ID 입니다."), 400

    pw_hash = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode()

    cur.execute(
        """
        INSERT INTO users (id, name, email, password_hash, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, name, email, pw_hash, datetime.now().isoformat()),
    )

    db.commit()
    db.close()

    return jsonify(ok=True, message="회원가입 완료!")


@app.post("/api/login")
def api_login():
    data = request.get_json() or {}
    user_id = data.get("id", "").strip()
    pw = data.get("password", "").strip()

    if not user_id or not pw:
        return jsonify(ok=False, message="ID와 비밀번호를 모두 입력해주세요."), 400

    db = get_db()
    cur = db.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    db.close()

    if row is None:
        return jsonify(ok=False, message="존재하지 않는 ID입니다."), 400

    pw_hash = row["password_hash"].encode("utf-8")

    if not bcrypt.checkpw(pw.encode("utf-8"), pw_hash):
        return jsonify(ok=False, message="비밀번호가 올바르지 않습니다."), 400

    session.clear()
    session["user_id"] = row["id"]
    session["user_name"] = row["name"]
    session["user_email"] = row["email"]
    session.permanent = True

    return jsonify(ok=True, message="로그인 성공"), 200


@app.get("/api/me")
def api_me():
    user_id = session.get("user_id")

    if not user_id:
        return jsonify(ok=False, message="로그인이 필요합니다."), 401

    db = get_db()
    cur = db.execute("SELECT id, name, email FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()

    cur2 = db.execute(
        "SELECT player_name FROM saved_players WHERE user_id=?", (user_id,)
    )
    saved = [r["player_name"] for r in cur2.fetchall()]

    db.close()

    return (
        jsonify(
            ok=True,
            user={
                "id": row["id"],
                "name": row["name"],
                "email": row["email"],
                "saved_players": saved,
            },
        ),
        200,
    )


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify(ok=True, message="로그아웃 되었습니다."), 200


# ------------------------------------------------------------------
# 4. 선수 즐겨찾기 (찜)
# ------------------------------------------------------------------
@app.post("/api/favorite")
def api_favorite():
    """
    body: { "player_name": "Lionel Messi", "favorite": true/false }
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(ok=False, message="로그인이 필요합니다."), 401

    data = request.json or {}
    player_name = (data.get("player_name") or "").strip()
    favorite = bool(data.get("favorite"))

    if not player_name:
        return jsonify(ok=False, message="선수 이름이 필요합니다."), 400

    db = get_db()
    cur = db.cursor()

    if favorite:
        cur.execute(
            "SELECT 1 FROM saved_players WHERE user_id=? AND player_name=?",
            (user_id, player_name),
        )
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO saved_players (user_id, player_name) VALUES (?, ?)",
                (user_id, player_name),
            )
    else:
        cur.execute(
            "DELETE FROM saved_players WHERE user_id=? AND player_name=?",
            (user_id, player_name),
        )

    db.commit()
    db.close()

    return jsonify(ok=True, favorite=favorite), 200


# ------------------------------------------------------------------
# 5. 선수 목록 / 단일 선수 정보 (player_with_pred.xlsx)
# ------------------------------------------------------------------
@app.get("/api/players")
def api_players():
    try:
        load_players_df()
    except Exception as e:
        print("Excel 읽기 에러:", e)
        return jsonify(ok=False, message="선수 목록을 불러올 수 없습니다."), 500

    df = players_df
    name_col = get_name_col(df)
    if not name_col:
        return jsonify(ok=False, message="엑셀에 선수 이름 컬럼이 없습니다."), 500

    names = sorted(
        {
            str(n).strip()
            for n in df[name_col].dropna().tolist()
            if str(n).strip()
        }
    )

    return jsonify(ok=True, players=names), 200


@app.get("/api/player")
def api_player():
    raw_name = (request.args.get("name") or "").strip()
    if not raw_name:
        return jsonify(ok=False, message="선수 이름이 필요합니다."), 400

    try:
        load_players_df()
    except Exception as e:
        print("Excel 읽기 에러:", e)
        return jsonify(ok=False, message="선수 정보를 불러올 수 없습니다."), 500

    df = players_df
    name_col = get_name_col(df)
    if not name_col:
        return jsonify(ok=False, message="엑셀에 선수 이름 컬럼이 없습니다."), 500

    name_lower = raw_name.lower()
    mask = df[name_col].astype(str).str.strip().str.lower() == name_lower

    if not mask.any():
        return (
            jsonify(ok=False, message=f"'{raw_name}' 선수 정보를 찾을 수 없습니다."),
            404,
        )

    row = df[mask].iloc[0]

    birth = row.get("birth") or row.get("date of birth") or row.get("생년월일")
    position = row.get("position") or row.get("pos") or row.get("포지션")
    present_value = row.get("present market value") or row.get("present_value") or row.get("시장가치")
    predicted_value = row.get("predicted market value") or row.get("predicted_value") or row.get("예측시장가치")

    player_data = {
        "name": safe_str(row.get(name_col) or raw_name),
        "birth": safe_str(birth),
        "position": safe_str(position),
        "present_value": safe_str(present_value),
        "predicted_value": safe_str(predicted_value),
    }

    return jsonify(ok=True, player=player_data), 200


# ------------------------------------------------------------------
# 6. 모델 feature / 예측 API
#
#    /api/features는 내부적으로 학습에 쓰인 47개 엔지니어링 피처
#    (log_전시즌_시장가치, 나이_제곱 등)를 그대로 노출하지 않는다.
#    실사용자가 계산할 수 없는 값이기 때문이다. 대신 원본 스탯
#    (득점, 어시스트, 터치 ...)만 노출하고, /api/predict가 내부에서
#    features.engineer_single()로 나머지를 계산한다.
# ------------------------------------------------------------------
@app.get("/api/features")
def api_features():
    return jsonify(
        ok=True,
        positions=features.POSITION_GROUPS,
        stat_fields=features.RAW_STAT_FIELDS,
        change_tracked_stats=features.STATS_FOR_CHANGE,
    ), 200


@app.post("/api/predict")
def api_predict():
    if not MODELS_BY_POSITION:
        return jsonify(ok=False, message="모델이 로딩되어 있지 않습니다."), 500

    data = request.get_json() or {}
    position = (data.get("position") or "").strip()
    group = features.group_position_detailed(position)

    if group not in MODELS_BY_POSITION:
        return jsonify(ok=False, message=f"'{position}' 포지션에 대한 예측 모델이 없습니다."), 400

    def to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    age = to_float(data.get("age"))
    prev_market_value = to_float(data.get("prev_market_value"))
    injury_days = to_float(data.get("injury_days"))
    current_stats = {k: to_float(v) for k, v in (data.get("current_stats") or {}).items()}
    prev_stats = {k: to_float(v) for k, v in (data.get("prev_stats") or {}).items()}

    model_info = MODELS_BY_POSITION[group]
    X = features.engineer_single(
        position=position,
        age=age,
        prev_market_value=prev_market_value,
        injury_days=injury_days,
        current_stats=current_stats,
        prev_stats=prev_stats,
        feature_cols=model_info["feature_cols"],
    )

    try:
        log_pred = float(model_info["model"].predict(X)[0])
    except Exception as e:
        print("예측 중 에러:", e)
        return jsonify(ok=False, message="모델 예측 중 오류가 발생했습니다."), 500

    predicted_value = float(np.expm1(log_pred))
    return jsonify(ok=True, predicted_value=predicted_value, model_position=group), 200


# ------------------------------------------------------------------
# 7. 프런트엔드 정적 파일 서빙
# ------------------------------------------------------------------
@app.route("/")
@app.route("/index.html")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
