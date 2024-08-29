import pandas as pd
import random
import time
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from boruta import BorutaPy
from tqdm import tqdm
import joblib
import boruta


# 抑制特定警告
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")
balanced_samples_path_2023 = 'train_data/2023_CVE_CWE.csv'

balanced_samples = pd.read_csv(balanced_samples_path_2023, on_bad_lines='skip', engine='python')

print("前五行 (balanced_samples):")
print(balanced_samples.head())
print("數據框的基本訊息 (balanced_samples):")
print(balanced_samples.info())
print("資料集描述 (balanced_samples):")
print(balanced_samples.describe())


# 把標籤'P' 和 'N' 轉為數字
balanced_samples['P/N'] = balanced_samples['P/N'].replace({'P': 1, 'N': 0})

# 然後刪除有缺失值的行
balanced_samples = balanced_samples.dropna(subset=['P/N'])

# 最後將 'P/N' 列轉換為整數類型
balanced_samples['P/N'] = balanced_samples['P/N'].astype(int)

# 確認 'P/N' 標籤的分佈
print("P/N 標籤的分佈:")
print(balanced_samples['P/N'].value_counts())

# 將數據集拆分為訓練集和驗證集
train_texts = val_texts = balanced_samples[['CVE-Description', 'CWE-Description']]
train_labels = val_labels = balanced_samples['P/N']

# 打印訓練集和驗證集的大小
print("訓練集樣本數量:", train_texts.shape[0])
print("驗證集樣本數量:", val_texts.shape[0])

# 顯示訓練集和驗證集的大小(包含標籤)
print("訓練集大小:", train_texts.shape, "訓練標籤大小:", train_labels.shape)
print("驗證集大小:", val_texts.shape, "驗證標籤大小:", val_labels.shape)

# 顯示訓練集和驗證集標籤PN數量，確認轉換Label有無成功
print("\n訓練集標籤PN:\n", train_labels.value_counts())
print("驗證集標籤PN:\n", val_labels.value_counts())

# 顯示總共訓練和驗證集樣本查看，確保沒有轉換錯
print("\n訓練集樣本:\n", train_texts.head())
print("驗證集樣本:\n", val_texts.head())

# 使用vector將文本轉成向量化
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts['CVE-Description'] + ' ' + train_texts['CWE-Description'])
print("特徵數量:", X_train.shape[1])
X_val = vectorizer.transform(val_texts['CVE-Description'] + ' ' + val_texts['CWE-Description'])

# 使用 Boruta 算法進行特徵選擇
forest = RandomForestClassifier(n_jobs=-1, max_depth=5)
boruta_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)

start_time = time.time()  # 記錄開始時間

# 初始化隨機森林分類器
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

train_losses = []
val_losses = []

# 計算正樣本數量並初始化批次大小
num_positive_samples = len(train_labels[train_labels == 1])
batch_size = num_positive_samples

for i in tqdm(range(0, X_train.shape[0], batch_size), desc="Processing batches"):
    batch_X_train = X_train[i:i + batch_size].toarray()
    batch_train_labels = train_labels.iloc[i:i + batch_size].values

    # 特徵選擇
    boruta_selector.fit(batch_X_train, batch_train_labels)

    # 訓練隨機森林模型
    clf.fit(batch_X_train[:, boruta_selector.support_], batch_train_labels)

    # 計算訓練損失
    train_preds = clf.predict(batch_X_train[:, boruta_selector.support_])
    train_loss = 1 - accuracy_score(batch_train_labels, train_preds)
    train_losses.append(train_loss)

    # 計算驗證損失
    val_preds = clf.predict(X_val[:, boruta_selector.support_])
    val_loss = 1 - accuracy_score(val_labels, val_preds)
    val_losses.append(val_loss)

end_time = time.time()  # 記錄結束時間

# 預測
val_preds = clf.predict(X_val[:, boruta_selector.support_])

# 評估模型
accuracy = accuracy_score(val_labels, val_preds)
precision = precision_score(val_labels, val_preds)
recall = recall_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 訓練所花費的時間
training_time = end_time - start_time
print(f"訓練時間: {training_time:.2f} 秒")

# 計算並打印平均損失
average_train_loss = np.mean(train_losses)
average_val_loss = np.mean(val_losses)
print(f"平均訓練損失: {average_train_loss:.4f}")
print(f"平均驗證損失: {average_val_loss:.4f}")

# 保存模型
joblib.dump(clf, 'train_model_save_cwe/random_forest_model_cvecwe.joblib')
print("隨機森林模型已保存")

# 保存特徵選擇器
joblib.dump(boruta_selector, 'train_model_save_cwe/boruta_selector_cvecwe.joblib')
print("特徵選擇器已保存")

# 保存向量化器
joblib.dump(vectorizer, 'train_model_save_cwe/vectorizer_cvecwe.joblib')
print("向量化器已保存")
