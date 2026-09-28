"""
Transfermarkt 시장 가치 변동 이력 수집.

원래 `code/marketvalue_all.ipynb`는 `get_market_value_history()` 함수를
정의만 해두고 실제로는 한 번도 호출하지 않는 죽은 코드였고(선수 목록도
프리미어리그로 하드코딩), `code/marketvalue.ipynb`는 선수 1명(Cristian
Romero)만 테스트해보는 프로토타입이었다. 이 스크립트는 그 두 개를 합쳐
`data/raw/transfermarkt_player_ids_<league>.csv`에 있는 모든 선수에 대해
실제로 루프를 도는 버전이다.

실행 예시:
    python fetch_market_values.py --league premier --out market_values_premier.csv
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    )
}


def get_market_value_history(player_id: str) -> list[dict]:
    url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    return [
        {
            "Transfermarkt_ID": player_id,
            "날짜": datetime.fromtimestamp(item["x"] / 1000).strftime("%Y-%m-%d"),
            "시장가치_EUR": item["y"],
            "소속팀": item["verein"],
            "나이": item["age"],
        }
        for item in resp.json().get("list", [])
    ]


def fetch_league(league: str, out_path: Path) -> None:
    ids_path = DATA_RAW / f"transfermarkt_player_ids_{league}.csv"
    if not ids_path.exists():
        print(f"건너뜀: {ids_path} 없음")
        return

    id_df = pd.read_csv(ids_path)
    all_rows = []
    for _, row in id_df.iterrows():
        try:
            history = get_market_value_history(str(row["Transfermarkt_ID"]))
            for h in history:
                h["선수명"] = row["선수명"]
            all_rows.extend(history)
        except requests.exceptions.RequestException as exc:
            print(f"[오류] {row['선수명']} ({row['Transfermarkt_ID']}): {exc}")
        time.sleep(0.5)

    pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"{league}: 선수 {len(id_df)}명, 기록 {len(all_rows)}건 -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fetch_league(args.league, Path(args.out))
