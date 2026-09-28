import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from tqdm import tqdm
import os

def search_transfermarkt_player_by_name(player_name):
    """변경된 HTML 구조에 맞춰 선수 ID를 검색하고 반환합니다."""
    
    search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
    params = {'query': player_name}
    
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        first_result = soup.select_one("td.hauptlink a")
        
        if first_result:
            href = first_result.get('href', '')
            player_id_match = re.search(r'/spieler/(\d+)', href)
            if player_id_match:
                return player_id_match.group(1)
        
        return None
        
    except requests.exceptions.RequestException as e:
        # 네트워크 오류가 발생하면 즉시 메시지를 출력합니다.
        tqdm.write(f"-> '{player_name}' 선수 검색 중 네트워크 오류: {e}")
        return None

# --- 1단계: unique_players_list.csv 파일 읽기 ---
input_filename = 'unique_players_list_premier.csv'

if not os.path.exists(input_filename):
    print(f"❌ 오류: '{input_filename}' 파일이 없습니다. 이 파일을 먼저 생성해주세요.")
else:
    print(f"✅ 1단계: '{input_filename}' 파일에서 선수 명단을 읽어옵니다.")
    unique_players_df = pd.read_csv(input_filename)
    print(f"총 {len(unique_players_df)}명의 선수에 대한 ID 검색을 시작합니다.")

    # --- 2단계: 각 선수의 Transfermarkt ID 검색 및 저장 ---
    print("\n✅ 2단계: 각 선수의 Transfermarkt ID를 검색하며 실시간으로 결과를 출력합니다.")
    player_id_data = []
    
    for _, row in tqdm(unique_players_df.iterrows(), total=unique_players_df.shape[0], desc="ID 검색 중"):
        player_name = row['선수명']
        player_id = search_transfermarkt_player_by_name(player_name)
        
        # --- 여기가 수정된 부분 ---
        # ID를 찾았는지 여부에 따라 즉시 다른 메시지를 출력합니다.
        if player_id:
            # 성공한 경우, 데이터를 저장하고 성공 메시지를 바로 출력합니다.
            player_id_data.append({'선수명': player_name, 'Transfermarkt_ID': player_id})
            tqdm.write(f"✅ [성공] {player_name}: ID {player_id} 찾음")
        else:
            # 실패한 경우, 실패 메시지를 바로 출력합니다.
            tqdm.write(f"❌ [실패] '{player_name}' 선수의 ID를 찾지 못했습니다.")
        
        time.sleep(0.5)

    # --- 3단계: 최종 결과물 생성 (전체 작업 요약) ---
    if player_id_data:
        id_df = pd.DataFrame(player_id_data)
        
        output_filename = 'transfermarkt_player_ids.csv'
        id_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print("\n\n--- [ 최종 요약 ] ---")
        print(f"🎉 모든 작업 완료! 총 {len(id_df)}명의 선수 ID를 찾아 '{output_filename}' 파일에 저장되었습니다.")
        print("\n[ 저장된 결과 (상위 10개) ]")
        print(id_df.head(10).to_string())
    else:
        print("\n처리된 데이터가 없습니다. 선수 검색 결과를 확인해보세요.")