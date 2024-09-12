import os
import pandas as pd
import random
import nltk
from tqdm import tqdm

# 確保已經下載了 wordnet 資料庫
nltk.download('wordnet')

# 載入新檔案與CAPEC對應檔案
new_file_path = 'final_merged.csv'
file2_path = '../inference_data/CAPEC_to_CWE_Mapping_v3.csv'
new_file_df = pd.read_csv(new_file_path)
file2_df = pd.read_csv(file2_path)

# 使用CWE-ID作為鍵值將新檔案與CAPEC對應資料進行合併
new_merged_df = pd.merge(new_file_df, file2_df[['CWE-ID', 'CAPEC-ID', 'CAPEC-Description']], on='CWE-ID', how='left')

# 刪除沒有對應CAPEC資料的行
filtered_df = new_merged_df.dropna(subset=['CAPEC-ID'])

# 創建儲存結果的資料夾
# output_dir = ''
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# 讀取 CAPEC 描述數據
df_cwe = pd.read_csv('../inference_data/CAPEC_Desc_CWE_254.csv', encoding='utf-8')

# 將 CAPEC-ID 和 CWE-ID 映射解析為字典
capec_to_cwes = {}
for _, row in df_cwe.iterrows():
    capec_id = row['CAPEC-ID']
    cwes = set(map(int, row['CWE-ID'].split(',')))
    capec_to_cwes[capec_id] = cwes

# 使用列表收集所有的數據
rows = []

# 使用tqdm封裝以顯示進度條
positive_samples = 0
negative_samples = 0

for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="處理數據"):
    cwe_ids = set(map(int, row['CWE-ID'].split(','))) if isinstance(row['CWE-ID'], str) else {int(row['CWE-ID'])}
    rows.append({**row, 'P/N': 'P'})
    positive_samples += 1  # 計數正樣本

    negative_capec_ids = [capec for capec, cwes in capec_to_cwes.items() if not cwes.intersection(cwe_ids)]
    if negative_capec_ids:
        selected_capec = random.choice(negative_capec_ids)
        negative_description = df_cwe[df_cwe['CAPEC-ID'] == selected_capec]['CAPEC-Description'].iloc[0]
        rows.append({**row, 'CAPEC-ID': selected_capec, 'CAPEC-Description': negative_description, 'P/N': 'N'})
        negative_samples += 1  # 計數負樣本
    else:
        print(f"未找到適合的負樣本 CWE-ID: {row['CWE-ID']} 於索引 {index}")

# 保存數據結果
df_final = pd.DataFrame(rows)
output_path = 'for_cvecapec_3gpp.csv'
df_final.to_csv(output_path, index=False, encoding='utf-8')

print(f"數據已保存到 {output_path}, 共有 {df_final.shape[0]} 行數據")
print(f"正樣本數: {positive_samples}, 負樣本數: {negative_samples}")
