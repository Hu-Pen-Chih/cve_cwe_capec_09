import pandas as pd
import os

# 設定CSV檔案的目錄
directory = '3GPP_SA3_篩選完句子的檔案_0820'

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

# 將合併、去重並排序後的資料寫入新的CSV檔案
output_file_path = os.path.join(directory, 'final_merged_output0819-1.csv')
try:
    merged_df.to_csv(output_file_path, index=False)
    print(f"Merged, deduplicated, and sorted file saved to {output_file_path}")
except Exception as e:
    print(f"Error saving merged file: {e}")
