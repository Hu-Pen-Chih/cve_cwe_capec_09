import os
import pandas as pd

# 設定CSV檔案的目錄
directory = '3GPP_SA2_final_inference'

# 初始化空的DataFrame
merged_df = pd.DataFrame()

# 遍歷目錄中的每個檔案
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # 只處理CSV檔案
        file_path = os.path.join(directory, filename)
        try:
            df = pd.read_csv(file_path)
            df = df.dropna()  # 移除包含空值的行
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# 移除 `Sentence` 欄位重複的行，只保留一個
merged_df = merged_df.drop_duplicates(subset='Sentence')

# 對合併後的DataFrame進行排序，按照Positive Probability欄位由高到低
merged_df = merged_df.sort_values(by='Positive Probability', ascending=False)

# 計算不同範圍內的行數
count_99 = len(merged_df[merged_df['Positive Probability'] > 0.99])
count_991 = len(merged_df[merged_df['Positive Probability'] > 0.991])
count_992 = len(merged_df[merged_df['Positive Probability'] > 0.992])
count_993 = len(merged_df[merged_df['Positive Probability'] > 0.993])
count_994 = len(merged_df[merged_df['Positive Probability'] > 0.994])
count_995 = len(merged_df[merged_df['Positive Probability'] > 0.995])
count_996 = len(merged_df[merged_df['Positive Probability'] > 0.996])
count_997 = len(merged_df[merged_df['Positive Probability'] > 0.997])
count_998 = len(merged_df[merged_df['Positive Probability'] > 0.998])
count_999 = len(merged_df[merged_df['Positive Probability'] > 0.999])

# 顯示每個範圍的行數
print(f"Positive Probability > 0.99: {count_99} rows")
print(f"Positive Probability > 0.991: {count_991} rows")
print(f"Positive Probability > 0.992: {count_992} rows")
print(f"Positive Probability > 0.993: {count_993} rows")
print(f"Positive Probability > 0.994: {count_994} rows")
print(f"Positive Probability > 0.995: {count_995} rows")
print(f"Positive Probability > 0.996: {count_996} rows")
print(f"Positive Probability > 0.997: {count_997} rows")
print(f"Positive Probability > 0.998: {count_998} rows")
print(f"Positive Probability > 0.999: {count_999} rows")

if 'CWE-ID' in merged_df.columns:
    cwe_counts = merged_df['Mapped CWE-ID'].value_counts()
    print("\n所有不同的 CWE-ID 及其對應的數量:")
    for cwe_id, count in cwe_counts.items():
        print(f"CWE-ID: {cwe_id}, 數量: {count}")
else:
    print("CWE-ID 欄位未找到，無法列出 CWE-ID。")

# 更改欄位名稱
merged_df.rename(columns={
    'File Name': 'CVE-ID',
    'Sentence': 'CVE-Description',
    'Mapped CWE-ID': 'CWE-ID',
    'CWE Description': 'CWE-Description'
}, inplace=True)

# 只保留 Positive Probability > 0.99 的資料並另存為新檔案
filtered_df = merged_df[merged_df['Positive Probability'] > 0.991]

filtered_output_file_path = 'final_filtered_step4/final_filtered_prob_991_sa2.csv'
try:
    filtered_df.to_csv(filtered_output_file_path, index=False)
    print(f"Filtered file with Positive Probability > 0.991 saved to {filtered_output_file_path}")
except Exception as e:
    print(f"Error saving filtered file: {e}")
