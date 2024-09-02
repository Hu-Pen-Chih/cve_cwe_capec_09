import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


# 讀取 CSV 文件
file_path = 'All_CWE.csv'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

print("訓練資料下載成功")

# 提取特徵並刪除不必要的欄位
selected_features = ['CWE-ID', 'CWE-Description']
df_selected = df[selected_features]
df_selected = df_selected.drop_duplicates(subset=['CWE-ID'], keep='first')

# 定義推理模型
class InferenceModel(nn.Module):
    def __init__(self, num_labels=2):
        super(InferenceModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        avg_pooling_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        diff = torch.abs(cls_embeddings - avg_pooling_embeddings)
        mul = cls_embeddings * avg_pooling_embeddings
        combined_features = torch.cat((diff, mul), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        return logits

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = InferenceModel(num_labels=2)
device = torch.device('cpu')
model.load_state_dict(torch.load('final_model_average.pt', map_location=device))
#model.load_state_dict(torch.load('../../bert_capec/inference_model_v1/2023_0729/best_model_2023_0728_fully_valid.pt', map_location=device))
model.eval()

# 創建保存篩選結果的資料夾
output_folder = '3GPP_SA3_篩選完句子的檔案_0820'
os.makedirs(output_folder, exist_ok=True)

# 遍歷3GPP_SA3資料夾中的每個.txt檔案
folder_path = '3GPP_SA3_all'
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
all_high_prob_results = []  # 用於收集所有文件的高概率結果
 
for file_name in tqdm(file_list, desc='Overall progress'):
    output_file_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_high_prob.csv')
    if os.path.exists(output_file_path):
        print(f"{file_name} 已處理過，跳過")
        continue
    
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        high_prob_results = []

        for idx, line in tqdm(enumerate(lines), total=len(lines), desc=f'Processing {file_name}'):
            line = line.strip()
            if line:
                # Processing each CWE for the sentence
                tqdm_desc = f"Processing CWEs for sentence {idx+1}/{len(lines)}"
                with tqdm(total=len(df_selected), desc=tqdm_desc) as pbar:
                    max_prob = float('-inf')
                    best_cwe_id = None
                    best_cwe_description = None

                    for _, row in df_selected.iterrows():
                        cwe_id = row['CWE-ID']
                        cwe_description = row['CWE-Description']
                        combined_description = line + " [SEP] " + cwe_description
                        encoded_combined = tokenizer(combined_description, return_tensors='pt', padding=True, truncation=True, max_length=512)

                        with torch.no_grad():
                            combined_logits = model(encoded_combined['input_ids'], encoded_combined['attention_mask'])
                            combined_probabilities = F.softmax(combined_logits, dim=1)
                            positive_prob = combined_probabilities[:, 1].item()

                        if positive_prob > max_prob:
                            max_prob = positive_prob
                            best_cwe_id = int(cwe_id)
                            best_cwe_description = cwe_description
                        pbar.update(1)  # 更新進度條

                    if max_prob > 0.997:
                        result = {
                            "File Name": file_name,
                            "Sentence": line,
                            "Mapped CWE-ID": best_cwe_id,
                            "CWE Description": best_cwe_description,
                            "Positive Probability": max_prob
                        }
                        high_prob_results.append(result)
                        all_high_prob_results.append(result)

        # 保存每個文件的高概率結果到 CSV 文件
        if high_prob_results:
            high_prob_results_df = pd.DataFrame(high_prob_results)
            high_prob_results_df.to_csv(output_file_path, index=False, encoding='utf-8')

# 保存所有文件的高概率結果到一個 CSV 文件
if all_high_prob_results:
    all_high_prob_results_df = pd.DataFrame(all_high_prob_results)
    all_high_prob_results_df.to_csv('all_high_prob_sentences_0.99.csv', index=False, encoding='utf-8')

print("處理完成，所有高概率結果已保存到指定的資料夾和 all_high_prob_sentences_0.99.csv")