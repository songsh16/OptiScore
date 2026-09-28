"""
FotMob 선수 시즌 스탯 크롤러 (Selenium).

원래는 리그/시즌 조합마다(`chrolling/fotmob_premier_2016-2020_all.ipynb` 등)
같은 코드를 6~7벌 복붙해서 하드코딩된 league_id/season_id로 돌렸었다.
이 스크립트는 그걸 함수 하나로 정리한 버전이다. league_id/season_id는
FotMob 리그 페이지 URL(`/leagues/{league_id}/stats/season/{season_id}/...`)에서 확인한다.

실행 예시:
    python scrape_fotmob.py --league-id 47 --season-id 14022 --out fotmob_premier_2019_2020.csv

주의: 실제 서비스에 쓰인 데이터셋(../../data/raw/fotmob_*.xlsx)은 이 스크립트를
여러 리그·시즌 조합에 대해 반복 실행해서 모은 결과를 병합한 것이다.
크롬드라이버 경로와 FotMob의 CSS 클래스명은 시점에 따라 바뀔 수 있어
재실행 시 선택자를 다시 확인해야 할 수 있다.
"""

import argparse
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

STAT_NAMES = [
    "PK 제외한 xG", "xG 유효 슈팅 (xGOT)", "가로채기", "경고", "공격 지역 점유율",
    "공중 볼 경합 성공", "공중 볼 경합 성공 %", "기회 창출", "드리블 성공", "드리블로 제침",
    "득점", "롱 패스 정확도", "막힘", "반칙", "볼 경합 성공", "볼 경합 성공 %",
    "볼 뺏김", "상대편 박스 내에서의 터치", "성공한 크로스", "성공한 패스", "슛",
    "어시스트", "예상 골 (xG)", "예상 어시스트 (xA)", "유효 슈팅", "정확한 긴 패스",
    "크로스 정확도", "태클 성공", "태클 성공 %", "터치", "퇴장", "패스 정확도",
    "페널티 득점", "페널티킥 받음", "페널티킥 허용", "회복", "획득한 파울",
]


def scrape_season(league_id: int, season_id: int, driver_path: str) -> pd.DataFrame:
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(driver_path), options=options)
    wait = WebDriverWait(driver, 20)

    all_rows = []
    visited = set()
    page_num = 0

    try:
        while True:
            url = (
                f"https://www.fotmob.com/ko/leagues/{league_id}/stats/season/"
                f"{season_id}/players/rating?page={page_num}"
            )
            driver.get(url)
            time.sleep(3)

            try:
                wait.until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "a[href^='/ko/players/']")
                    )
                )
            except Exception:
                break

            links = [
                a.get_attribute("href")
                for a in driver.find_elements(By.CSS_SELECTOR, "a[href^='/ko/players/']")
            ]
            new_links = [u for u in links if u not in visited]
            visited.update(new_links)

            for player_url in new_links:
                row = _scrape_player(driver, wait, player_url)
                if row:
                    all_rows.append(row)

            page_num += 1
    finally:
        driver.quit()

    return pd.DataFrame(all_rows)


def _scrape_player(driver, wait, player_url: str) -> dict | None:
    try:
        driver.get(player_url)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "css-zt63wq-PlayerNameCSS")))
        time.sleep(2)

        try:
            player_name = driver.find_element(By.CLASS_NAME, "css-zt63wq-PlayerNameCSS").text.strip().lower()
        except Exception:
            player_name = player_url.split("/")[-1].replace("-", " ")

        # "90분당" (per-90) 필터로 전환
        buttons = driver.find_elements(By.CLASS_NAME, "css-1efq0w6-FilterButton")
        per90_button = next((b for b in buttons if b.text.strip() == "90분당"), None)
        if per90_button is None:
            return None
        per90_button.click()
        time.sleep(3)

        titles = driver.find_elements(By.CLASS_NAME, "css-2duihq-StatTitle")
        values = driver.find_elements(By.CLASS_NAME, "css-jb6lgd-StatValue")
        stats = {
            t.text.strip(): v.text.strip()
            for t, v in zip(titles, values)
            if t.text.strip() in STAT_NAMES
        }

        row = {key: stats.get(key, "-") for key in STAT_NAMES}
        row["선수명"] = player_name
        row["URL"] = player_url
        return row
    except Exception as exc:
        print(f"[오류] {player_url}: {exc}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--season-id", type=int, required=True)
    parser.add_argument("--driver-path", default="C:/chromedriver-win64/chromedriver-win64/chromedriver.exe")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = scrape_season(args.league_id, args.season_id, args.driver_path)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"완료: {len(df)}명, {args.out}에 저장")
