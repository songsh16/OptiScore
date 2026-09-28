"""
FotMob 선수명 -> Transfermarkt 선수 ID 매칭.

`code/id_bundesliga.ipynb`, `id_champs.ipynb`, `id_laliga.ipynb`, `id_ligue1.ipynb`,
`id_premier.ipynb`, `id_serieA.ipynb` 6개가 리그 이름만 다르고 완전히 동일한
로직을 복붙해서 갖고 있던 것을 함수 하나로 통합했다.

입력: data/raw/unique_players_list_<league>.csv (컬럼: 선수명)
출력: data/raw/transfermarkt_player_ids_<league>.csv (컬럼: 선수명, Transfermarkt_ID)

실행 예시:
    python match_transfermarkt_ids.py --league premier
    python match_transfermarkt_ids.py --league bundesliga champs laliga ligue1 premier serieA
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    )
}


def search_transfermarkt_id(player_name: str) -> str | None:
    url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    try:
        resp = requests.get(url, headers=HEADERS, params={"query": player_name}, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        tqdm.write(f"-> '{player_name}' 검색 중 네트워크 오류: {exc}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    first_result = soup.select_one("td.hauptlink a")
    if not first_result:
        return None

    match = re.search(r"/spieler/(\d+)", first_result.get("href", ""))
    return match.group(1) if match else None


def match_league(league: str) -> None:
    input_path = DATA_RAW / f"unique_players_list_{league}.csv"
    output_path = DATA_RAW / f"transfermarkt_player_ids_{league}.csv"

    if not input_path.exists():
        print(f"건너뜀: {input_path} 없음")
        return

    players = pd.read_csv(input_path)
    matched, failed = [], []

    for _, row in tqdm(players.iterrows(), total=len(players), desc=f"{league} ID 검색"):
        name = row["선수명"]
        player_id = search_transfermarkt_id(name)
        if player_id:
            matched.append({"선수명": name, "Transfermarkt_ID": player_id})
        else:
            failed.append(name)
        time.sleep(0.5)  # 요청 과다로 차단되지 않도록 간격 유지

    pd.DataFrame(matched).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{league}: {len(matched)}명 매칭, {len(failed)}명 실패 -> {output_path}")
    if failed:
        print("  실패 목록:", ", ".join(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league", nargs="+", required=True,
        choices=["bundesliga", "champs", "laliga", "ligue1", "premier", "serieA"],
    )
    args = parser.parse_args()
    for league in args.league:
        match_league(league)
