import os
import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm

# 確保已經下載了 wordnet 資料
nltk.download('wordnet')

# 定義增強次數變數
AUGMENT_TIMES = 1 
# 文件路徑
cve_file_path = 'final_merged.csv'
all_cwe_file_path = '../inference_data/All_CWE.csv'

# # 創建儲存結果的資料夾
# output_dir = 'tagging_3gpp_data'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# 讀取 CSV 文件的函數
def load_csv_with_fallback(file_path, use_cols=None):
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

# 讀取數據
df_cwe = load_csv_with_fallback(all_cwe_file_path, use_cols=['CWE-ID', 'CWE-Description'])
df_selected = load_csv_with_fallback(cve_file_path, use_cols=['CVE-ID', 'CWE-ID', 'CVE-Description', 'CWE-Description'])

# 創建CVE-ID到CWE-ID的字典
cve_to_cwe_dict = df_selected.set_index('CVE-ID')['CWE-ID'].to_dict()

# 同義詞替換函數
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, n=2):
    words = sentence.split()
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

# 使用列表收集所有的數據
rows_synonyms = []
rows_repeated = []
common_negative_samples = []

# 取得所有的CWE-ID
all_cwe_ids = df_cwe['CWE-ID'].unique()

# tqdm 封裝以顯示進度條
for cve_id, cve_description in tqdm(zip(df_selected['CVE-ID'], df_selected['CVE-Description']), total=df_selected.shape[0], desc="處理CVE條目"):
    cwe_id = cve_to_cwe_dict.get(cve_id)
    cwe_description = df_selected[df_selected['CWE-ID'] == cwe_id]['CWE-Description'].iloc[0]

    # 重複正樣本
    for _ in range(AUGMENT_TIMES):  
        rows_repeated.append({'CVE-ID': cve_id, 'CVE-Description': cve_description, 'CWE-ID': cwe_id, 'CWE-Description': cwe_description, 'P/N': 'P'})

    # 同義詞轉換
    rows_synonyms.append({'CVE-ID': cve_id, 'CVE-Description': cve_description, 'CWE-ID': cwe_id, 'CWE-Description': cwe_description, 'P/N': 'P'})
    for _ in range(AUGMENT_TIMES - 1): # 原始樣本 + 同義詞轉換樣本
        augmented_cve_description = synonym_replacement(cve_description, n=2)
        rows_synonyms.append({'CVE-ID': cve_id, 'CVE-Description': augmented_cve_description, 'CWE-ID': cwe_id, 'CWE-Description': cwe_description, 'P/N': 'P'})

    # 隨機選取負樣本
    cwe_ids = list(all_cwe_ids)
    if cwe_id in cwe_ids:
        cwe_ids.remove(cwe_id)
    random.shuffle(cwe_ids)

    negative_samples_for_this_cve = []
    for random_cwe_id in cwe_ids[:AUGMENT_TIMES]:  # 選取 AUGMENT_TIMES 個負樣本
        random_cwe_description = df_cwe[df_cwe['CWE-ID'] == random_cwe_id]['CWE-Description'].iloc[0]
        negative_sample = {'CVE-ID': cve_id, 'CVE-Description': cve_description, 'CWE-ID': random_cwe_id, 'CWE-Description': random_cwe_description, 'P/N': 'N'}
        common_negative_samples.append(negative_sample)
        negative_samples_for_this_cve.append(negative_sample)
    
    rows_repeated.extend(negative_samples_for_this_cve)
    rows_synonyms.extend(negative_samples_for_this_cve)

# 創建DataFrame
df_mapping_repeated = pd.DataFrame(rows_repeated)
df_mapping_synonyms = pd.DataFrame(rows_synonyms)

# 保存CSV檔案到指定資料夾中
#df_mapping_repeated.to_csv(os.path.join(output_dir, f'for_cvecwe_3gpp_4k.csv'), index=False, encoding='utf-8')
#df_mapping_synonyms.to_csv(os.path.join(output_dir, f'synonym_samples_{AUGMENT_TIMES}_times_0815_your.csv'), index=False, encoding='utf-8')
df_mapping_repeated.to_csv('for_cvecwe_3gpp_4k.csv', index=False, encoding='utf-8')


# 列印數據和標籤統計
print("重複正樣本的CSV檔案行數:", df_mapping_repeated.shape[0])
print("同義詞轉換的CSV檔案行數:", df_mapping_synonyms.shape[0])
print("\n重複正樣本的P/N標籤數量分佈:\n", df_mapping_repeated['P/N'].value_counts())
print("同義詞轉換的P/N標籤數量分佈:\n", df_mapping_synonyms['P/N'].value_counts())

# 列印前五個樣本
print("\n重複正樣本的前五個CVE:")
print(df_mapping_repeated[['CVE-ID', 'CVE-Description', 'CWE-ID', 'CWE-Description', 'P/N']].head(5))

print("\n同義詞轉換的前五個CVE:")
print(df_mapping_synonyms[['CVE-ID', 'CVE-Description', 'CWE-ID', 'CWE-Description', 'P/N']].head(5))
