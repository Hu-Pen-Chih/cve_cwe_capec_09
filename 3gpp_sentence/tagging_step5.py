import os
import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm

# 確保已經下載了 wordnet 資料
nltk.download('wordnet')

# === 參數設置 ===
# 定義最少樣本數量與增強次數變數
MIN_SAMPLES = 25  # 每個CWE最少要有25個樣本
AUGMENT_TIMES = 1  # 控制正樣本與負樣本生成次數
MAX_SAMPLES = 500  # 對於樣本數量超過300的CWE-ID，篩選300個樣本

# 文件路徑
cve_file_path_sa1 = 'final_filtered_prob_991_sa1.csv'
cve_file_path_sa2 = 'final_filtered_prob_991_sa2.csv'
cve_file_path_sa3 = 'final_filtered_prob_991_sa3.csv'
all_cwe_file_path = '../inference_data/All_CWE.csv'

# 檔案命名的動態變數
output_file_name = f'for_cvecwe_3gpp_{MAX_SAMPLES}over_{MIN_SAMPLES}min_{AUGMENT_TIMES}aug.csv'

# === 函數區塊 ===
# 讀取 CSV 文件的函數
def load_csv_with_fallback(file_path, use_cols=None):
    """
    嘗試用不同編碼讀取 CSV 檔案，直到成功為止。
    """
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    for encoding in encodings:
        try:
            print(f"嘗試使用 {encoding} 編碼讀取文件 {file_path}")
            df = pd.read_csv(file_path, encoding=encoding, usecols=use_cols)
            print(f"成功使用 {encoding} 編碼讀取文件 {file_path}")
            return df
        except Exception as e:
            print(f"使用 {encoding} 編碼讀取文件失敗: {e}")
    raise Exception("無法讀取文件，請檢查文件格式是否正確")

# 同義詞替換函數
def get_synonyms(word):
    """
    獲取一個單字的所有同義詞。
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# 根據句子長度決定替換單字的數量
def synonym_replacement(sentence):
    """
    替換句子中的部分單字為其同義詞。
    """
    words = sentence.split()
    length = len(words)
    
    # 根據句子長度決定替換多少個單字
    if length <= 10:
        n = 2  # 替換2個單字
    elif 10 < length <= 20:
        n = 3  # 替換3個單字
    else:
        n = 4  # 替換4個單字

    new_sentence = words[:]
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_sentence = [synonym if word == random_word else word for word in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_sentence)

# === 數據讀取與合併區塊 ===
# 1. 讀取數據並合併
df_cwe = load_csv_with_fallback(all_cwe_file_path, use_cols=['CWE-ID', 'CWE-Description'])

# 讀取 sa2 和 sa3 的 CSV 文件
df_sa1 = load_csv_with_fallback(cve_file_path_sa1, use_cols=['CVE-ID', 'CWE-ID', 'CVE-Description', 'CWE-Description', 'Positive Probability'])
df_sa2 = load_csv_with_fallback(cve_file_path_sa2, use_cols=['CVE-ID', 'CWE-ID', 'CVE-Description', 'CWE-Description', 'Positive Probability'])
df_sa3 = load_csv_with_fallback(cve_file_path_sa3, use_cols=['CVE-ID', 'CWE-ID', 'CVE-Description', 'CWE-Description', 'Positive Probability'])

# 合併 sa2 和 sa3 的數據
df_selected = pd.concat([df_sa1, df_sa2, df_sa3], ignore_index=True)

# 打印每個CWE-ID的數量
print("處理前每個CWE-ID的樣本數量分佈:")
print(df_selected['CWE-ID'].value_counts())

# === 樣本處理與增強區塊 ===
# 2. 補充少於 MIN_SAMPLES 個樣本的 CWE-ID
rows_combined = []  # 包含原始樣本和增強樣本的最終結果
rows_synonyms = []  # 用來存放同義詞增強的樣本

for cwe_id, group in df_selected.groupby('CWE-ID'):
    cwe_description = group['CWE-Description'].iloc[0]
    
    # 將原始樣本保留
    if len(group) > MAX_SAMPLES:
        # 如果某個CWE的樣本數超過 MAX_SAMPLES，則按Positive Probability排序，取前MAX_SAMPLES個樣本
        group = group.sort_values(by='Positive Probability', ascending=False).head(MAX_SAMPLES)
    
    for idx, row in group.iterrows():
        rows_combined.append({
            'CVE-ID': row['CVE-ID'], 
            'CVE-Description': row['CVE-Description'], 
            'CWE-ID': cwe_id, 
            'CWE-Description': cwe_description, 
            'P/N': 'P',
            'Positive Probability': row['Positive Probability']
        })
    
    # 如果某個 CWE-ID 的樣本數量不足 MIN_SAMPLES 個，進行補充
    current_count = len(group)
    if current_count < MIN_SAMPLES:
        samples_needed = MIN_SAMPLES - current_count
        print(f"CWE-ID: {cwe_id} 需要增強 {samples_needed} 個樣本")

        # 用同義詞替換進行增強直到樣本數達到 MIN_SAMPLES
        for _ in range(samples_needed):
            random_row = group.sample(n=1).iloc[0]
            augmented_cve_description = synonym_replacement(random_row['CVE-Description'])
            rows_synonyms.append({
                'CVE-ID': random_row['CVE-ID'], 
                'CVE-Description': augmented_cve_description, 
                'CWE-ID': cwe_id, 
                'CWE-Description': cwe_description, 
                'P/N': 'P',
                'Positive Probability': random_row['Positive Probability']
            })

# 3. 合併原始樣本和增強樣本
rows_combined.extend(rows_synonyms)

# === 輸出合併後的 CWE-ID 總樣本數量分佈 ===
print("\n合併後的 CWE-ID 的樣本數量分佈:")
df_combined = pd.DataFrame(rows_combined)
print(df_combined['CWE-ID'].value_counts())

# === 正負樣本生成區塊 ===
# 4. 生成正樣本與負樣本（這裡應用 AUGMENT_TIMES）
all_cwe_ids = df_cwe['CWE-ID'].unique()

rows_with_negatives = []
for row in tqdm(rows_combined, desc="生成負樣本和正樣本增強"):
    # 重複生成正樣本
    for _ in range(AUGMENT_TIMES):
        rows_with_negatives.append(row)
    
    # 隨機選取負樣本
    cwe_id = row['CWE-ID']
    cwe_ids = list(all_cwe_ids)
    if cwe_id in cwe_ids:
        cwe_ids.remove(cwe_id)
    random.shuffle(cwe_ids)
    
    # 生成負樣本
    for _ in range(AUGMENT_TIMES):
        random_cwe_id = cwe_ids.pop()
        random_cwe_description = df_cwe[df_cwe['CWE-ID'] == random_cwe_id]['CWE-Description'].iloc[0]
        negative_sample = {
            'CVE-ID': row['CVE-ID'],
            'CVE-Description': row['CVE-Description'],
            'CWE-ID': random_cwe_id,
            'CWE-Description': random_cwe_description,
            'P/N': 'N',
            'Positive Probability': row['Positive Probability']  # 使用原始樣本的 Positive Probability
        }
        rows_with_negatives.append(negative_sample)

# === 保存檔案區塊 ===
# 創建最終 DataFrame
df_final = pd.DataFrame(rows_with_negatives)

# 保存CSV檔案到指定資料夾中
df_final.to_csv(output_file_name, index=False, encoding='utf-8')

# === 結果輸出 ===
# 列印數據和標籤統計
print("最終CSV檔案行數:", df_final.shape[0])
print("最終P/N標籤數量分佈:\n", df_final['P/N'].value_counts())
