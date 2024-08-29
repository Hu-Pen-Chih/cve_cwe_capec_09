import pandas as pd
import warnings
import torch
from transformers import BertTokenizer, BertModel
import time
import torch # 引入PyTorch資料庫，用於深度學習
import torch.nn as nn # 引入PyTorch的神經網路Function
from transformers import BertModel, BertTokenizer # 引入transformers資料庫中的BertModel和BertTokenizer
from torch.utils.data import DataLoader, Dataset # 引入PyTorch的DataLoader和Dataset模組
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 抑制特定警告
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")

# 設置文件路徑
balanced_samples_path_3gpp = 'train_data/for_cvecwe_3gpp_4k.csv'
balanced_samples_path_2023 = 'train_dat/2023_CVE_CWE.csv'

# 讀取處理好的數據
balanced_samples_3gpp = pd.read_csv(balanced_samples_path_3gpp, on_bad_lines='skip', engine='python')
balanced_samples_2023 = pd.read_csv(balanced_samples_path_2023, on_bad_lines='skip', engine='python')

# 顯示讀取的數據以供檢查
print("前五行 (balanced_samples_3gpp):")
print(balanced_samples_3gpp.head())
print("數據框的基本訊息 (balanced_samples_3gpp):")
print(balanced_samples_3gpp.info())
print("資料集描述 (balanced_samples_3gpp):")
print(balanced_samples_3gpp.describe())

print("前五行 (balanced_samples_2023):")
print(balanced_samples_2023.head())
print("數據框的基本訊息 (balanced_samples_2023):")
print(balanced_samples_2023.info())
print("資料集描述 (balanced_samples_2023):")
print(balanced_samples_2023.describe())

# 合併兩個數據框
balanced_samples = pd.concat([balanced_samples_3gpp, balanced_samples_2023])

# 先替換 'P' 和 'N' 為數字
balanced_samples['P/N'] = balanced_samples['P/N'].replace({'P': 1, 'N': 0})

# 然後刪除有缺失值的行
balanced_samples = balanced_samples.dropna(subset=['P/N'])

# 最後將 'P/N' 列轉換為整數類型
balanced_samples['P/N'] = balanced_samples['P/N'].astype(int)

# 確認數據完整性
print("確認數據完整性:")
print(balanced_samples.isnull().sum())

# 顯示讀取的數據以供檢查
print("前五行 (balanced_samples):")
print(balanced_samples.head())

# 確認 'P/N' 標籤的分佈
print("P/N 標籤的分佈:")
print(balanced_samples['P/N'].value_counts())

# 初始化 BERT Tokenizer，使用預訓練的 'bert-base-uncased' 模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_texts = val_texts = balanced_samples[['CVE-Description', 'CWE-Description']]
train_labels = val_labels = balanced_samples['P/N']

# 顯示訓練集和驗證集的大小(包含標籤)
print("訓練集大小:", train_texts.shape, "訓練標籤大小:", train_labels.shape)
print("驗證集大小:", val_texts.shape, "驗證標籤大小:", val_labels.shape)

# 顯示訓練集和驗證集標籤PN數量，確認轉換Label有無成功
print("\n訓練集標籤PN:\n", train_labels.value_counts())
print("驗證集標籤PN:\n", val_labels.value_counts())

# 顯示幾個訓練和驗證集樣本查看，確保沒有轉換錯
print("\n訓練集樣本:\n", train_texts.head())
print("驗證集樣本:\n", val_texts.head())

# 把有關於tokenization轉換的warning關掉
import logging # 引入 logging 庫
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # 關閉transformers中的特定warning

# 定義文本預處理函數，將文本進行編碼
def encode_texts(tokenizer, texts):
    return tokenizer(texts['CVE-Description'].tolist(), texts['CWE-Description'].tolist(),
                     padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# 編碼訓練集和驗證集文本
train_encodings = encode_texts(tokenizer, train_texts)
val_encodings = encode_texts(tokenizer, val_texts)

# 輸出訓練和驗證編碼結果的詳細信息
print("訓練編碼的key：", train_encodings.keys())
print("驗證編碼的key：", val_encodings.keys())

# 檢查訓練input_ids和attention_mask的尺寸
# 通常torch.Size([180, 512])，前面數字是數量，後面是文本最大長度
print("訓練input_ids尺寸：", train_encodings['input_ids'].shape)
print("訓練attention_mask尺寸：", train_encodings['attention_mask'].shape)
print("驗證input_ids尺寸：", val_encodings['input_ids'].shape)
print("驗證attention_mask尺寸：", val_encodings['attention_mask'].shape)

# 定義BERT模型
class CustomBERTModel(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 2, num_labels)       

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 取出 [CLS] token 的嵌入
        # 使用average pooling優化
        avg_pooling_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        diff = torch.abs(cls_embeddings - avg_pooling_embeddings)
        mul = cls_embeddings * avg_pooling_embeddings

        combined_features = torch.cat((diff, mul), dim=1)
        combined_features = self.dropout(combined_features)

        # 分類
        logits = self.classifier(combined_features)
        return logits

# 數據加載和模型訓練的準備
class SecurityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        label_value = self.labels[idx]
        # 檢查標籤值是否在合理範圍內
        if label_value not in [0, 1]:
            return self.__getitem__((idx + 1) % len(self))  # 跳過無效標籤，取下一個數據
        item['labels'] = torch.tensor(label_value, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# 將訓練數據和標籤、驗證數據和標籤轉換為上面設定的SecurityDataset的樣式
train_dataset = SecurityDataset(train_encodings, train_labels.to_numpy())
val_dataset = SecurityDataset(val_encodings, val_labels.to_numpy())

# 確認訓練集和驗證集的樣本數量
print("訓練集樣本數量:", len(train_dataset))
print("驗證集樣本數量:", len(val_dataset))

# 使用 DataLoader 將測試和驗證集分批次加載，batch_size設定為16
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 初始化模型和優化器
model = CustomBERTModel(num_labels=2)
# 加載預訓練權重
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) # 使用Adam優化器來更新模型參數，學習率設置為 1e-5

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

def save_model(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, 'vwa-model-cwe.pt')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

# 统一保存路徑
model_save_path = 'train_model_save_cwe'

# 確認保存路徑存在
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# 初始化最佳模型的損失為一個很高的數值
best_loss = float('inf')

# 儲存每次 epoch 的評估模型指標
train_losses = []  # 訓練損失
eval_losses = []   # 驗證損失
accuracies = []    # 準確度
precisions = []    # 精確度
recalls = []       # 召回率
f1_scores = []     # F1分數

# 將模型移動到設備 (GPU 或 CPU)
model.to(device)
# 開始計時
start_time = time.time()

# 訓練模型
for epoch in range(5):  # 訓練過程 epoch 迭代 5次
    model.train()  # 設置模型為訓練模式
    total_train_loss = 0  # 剛開始訓練損失設定為 0

    for batch in tqdm(train_loader, desc=f"訓練階段 Epoch {epoch+1}/5"):
        input_ids = batch['input_ids'].to(device)  # 獲取輸入的 input_ids 並移動到設備
        attention_mask = batch['attention_mask'].to(device)  # 獲取輸入的 attention_mask 並移動到設備
        labels = batch['labels'].to(device)  # 獲取真實標籤並移動到設備

        optimizer.zero_grad()  # 清空前一次迭代的梯度
        outputs = model(input_ids, attention_mask)  # 透過 forward 前向傳播獲取模型輸出
        loss = nn.CrossEntropyLoss()(outputs, labels)  # 獲取輸出之後透過 CrossEntropy 計算損失
        loss.backward()  # 計算梯度
        optimizer.step()  # 更新模型參數

        total_train_loss += loss.item()  # 累加訓練損失

    average_train_loss = total_train_loss / len(train_loader)  # 計算平均訓練損失
    train_losses.append(average_train_loss)  # 紀錄訓練損失

    # 驗證階段
    model.eval()  # 設置模型為驗證模式
    total_eval_loss = 0  # 剛開始設定初始化驗證損失為 0
    all_preds = []  # 初始化所有預測值的列表
    all_labels = []  # 初始化所有真實標籤的列表

    # 禁用梯度計算並驗證數據集（節省內存和計算資源）
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"驗證階段 Epoch {epoch+1}/5"):
            input_ids = batch['input_ids'].to(device)  # 獲取輸入的 input_ids 並移動到設備
            attention_mask = batch['attention_mask'].to(device)  # 獲取輸入的 attention_mask 並移動到設備
            labels = batch['labels'].to(device)  # 獲取真實標籤並移動到設備

            outputs = model(input_ids, attention_mask)  # 通過模型進行 forward，獲取輸出 outputs
            loss = nn.CrossEntropyLoss()(outputs, labels)  # 計算當前 batch 的 CrossEntropy Loss
            total_eval_loss += loss.item()  # 將每個 batch 計算出來的 Loss 累加上去

            _, predicted = torch.max(outputs, dim=1)  # 使用 torch.max 獲取預測結果
            all_preds.extend(predicted.cpu().numpy())  # 將當前 batch 的預測結果放在 all_preds 中，最後驗證完才會輸出 print
            all_labels.extend(labels.cpu().numpy())  # 將當前 batch 的真實標籤放在 all_preds 中，最後驗證完才會輸出 print

    average_eval_loss = total_eval_loss / len(val_loader)  # 計算平均驗證損失
    eval_losses.append(average_eval_loss)  # 紀錄驗證損失

    # 計算並列印準確度、精確度、召回率和 F1 分數
    accuracy = accuracy_score(all_labels, all_preds)  # 計算 accuracy 準確度
    precision = precision_score(all_labels, all_preds, average='binary')  # 計算 precision 精確度
    recall = recall_score(all_labels, all_preds, average='binary')  # 計算 recall 召回率
    f1 = f1_score(all_labels, all_preds, average='binary')  # 計算 F1 分數

    # 輸出每次 epoch 的評估模型指標
    print(f"Epoch {epoch+1}, Average Training Loss: {average_train_loss:.4f}, Average Validation Loss: {average_eval_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# 更新最佳模型並保存模型
    if average_eval_loss < best_loss:
        best_loss = average_eval_loss  # 更新最佳驗證損失
        save_model(model, model_save_path)  # 保存模型
        print(f"模型已保存到 {model_save_path}，當前最佳驗證損失為: {best_loss:.4f}")
    else:
        print(f"當前最佳驗證損失為: {best_loss:.4f}")

# 訓練結束計時
end_time = time.time()
total_training_time = end_time - start_time
print(f"總訓練時間: {total_training_time:.2f} 秒")
