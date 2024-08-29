import json
import os
import subprocess
import torch
import torch.nn as nn
import pandas as pd
import time
import re
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
import torch.nn.functional as F
import requests

# 設置 KUBECONFIG 環境變量
os.environ['KUBECONFIG'] = '/home/joehu/.kube/config'

# NVD API Key
NVD_API_KEY = '8ec1f7dd-7991-45ea-b6bd-7a57476f31bd'
base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# 設置資料夾
storage_dir = 'nvd_cve_storage_v2'
result_dir = 'inference_result'

general_storage_dir = 'nvd_cve_storage_v2'
fixed_report_storage_dir = os.path.join(general_storage_dir, 'report')

# 確保資料夾存在
os.makedirs(storage_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

print("開始設定推理模型...")

# 定義 CVE 到 CWE 的推理模型
class InferenceModelCVEtoCWE(nn.Module):
    def __init__(self, num_labels=2):
        super(InferenceModelCVEtoCWE, self).__init__()
        self.bert = BertModel(BertConfig())
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

# 定義 CWE 到 CAPEC 的推理模型
class InferenceModelCWEtoCAPEC(nn.Module):
    def __init__(self, num_labels=2):
        super(InferenceModelCWEtoCAPEC, self).__init__()
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

# 加載 CVE 到 CWE 的模型和分詞器
tokenizer_cve_cwe = BertTokenizer.from_pretrained("bert-base-uncased")
model_cve_cwe = InferenceModelCVEtoCWE(num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cve_cwe.load_state_dict(torch.load('vwa_result/cve_cwe.pt', map_location=device))
model_cve_cwe.eval()

# 加載 CWE 到 CAPEC 的模型和分詞器
tokenizer_cwe_capec = BertTokenizer.from_pretrained("bert-base-uncased")
model_cwe_capec = InferenceModelCWEtoCAPEC(num_labels=2)
model_cwe_capec.load_state_dict(torch.load('vwa_result/cve_capec.pt', map_location=device))
model_cwe_capec.eval()

print("模型加載成功。")

print("開始加載並處理 CWE 資料集...")
file_path_cwe = 'inference_data/All_CWE.csv'
try:
    df_cwe = pd.read_csv(file_path_cwe, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_cwe = pd.read_csv(file_path_cwe, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df_cwe = pd.read_csv(file_path_cwe, encoding='latin1')

selected_features = ['CWE-ID', 'CWE-Description']
df_cwe_selected = df_cwe[selected_features]
df_cwe_selected = df_cwe_selected.drop_duplicates(subset=['CWE-ID'], keep='first')

print("CWE 資料集加載並處理成功。")
print(df_cwe_selected.head())

print("開始加載 CAPEC to CWE Mapping 資料...")
# 加載推理和準確率計算所需的兩個不同CSV文件
file_path_inference = 'inference_data/CAPEC_Desc_CWE_254.csv'  # 用於推理的CSV
file_path_mapping = 'inference_data/CWE_Desc_CAPEC_129.csv'  # 用於準確率計算的CSV

try:
    df_inference = pd.read_csv(file_path_inference, encoding='utf-8')
    print("推理用CAPEC_Desc_CWE_254.csv資料集加載成功！")
    print(df_inference.head())
except Exception as e:
    print(f"推理用CAPEC_Desc_CWE_254.csv資料集加載失敗: {e}")

try:
    df_mapping = pd.read_csv(file_path_mapping, encoding='utf-8')
    print("準確率計算用CWE_Desc_CAPEC_129.csv資料集加載成功！")
    print(df_mapping.head())
except Exception as e:
    print(f"準確率計算用CWE_Desc_CAPEC_129.csv資料集加載失敗: {e}")


# 確認 nvd_cve_storage 資料夾是否存在，不存在則創建
storage_dir = 'nvd_cve_storage_v2'
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

def extract_cwe_id(cwe_id_str):
    match = re.search(r'\d+', cwe_id_str)
    return int(match.group()) if match else None

def fetch_cve_details(cve_id, storage_dir, use_local_only=False, max_retries=3, backoff_factor=0.3):
    storage_file = os.path.join(storage_dir, f"{cve_id}.json")
    if os.path.exists(storage_file):
        print(f"{cve_id} 的資料已存在，從本地文件加載...")
        with open(storage_file, 'r') as file:
            return json.load(file)
    
    if use_local_only:
        # 如果只使用本地資料，且文件不存在，則直接返回 None
        return None

    # 從 NVD API 獲取資料的部分
    url = f"{base_url}?cveId={cve_id}"
    headers = {
        "apiKey": NVD_API_KEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'vulnerabilities' in data and len(data['vulnerabilities']) > 0:
                    cve_data = data['vulnerabilities'][0]['cve']
                    if 'weaknesses' in cve_data and len(cve_data['weaknesses']) > 0:
                        cwe_data = cve_data['weaknesses'][0]['description'][0]['value']
                        if cwe_data not in ["NVD-CWE-Other", "NVD-CWE-noinfo", "Unknown"]:
                            cwe_id_str = cwe_data
                            cwe_id = extract_cwe_id(cwe_id_str)
                            result = {"CVE-ID": cve_id, "description": cve_data['descriptions'][0]['value'], "CWE-ID": cwe_id_str}
                            with open(storage_file, 'w') as file:
                                json.dump(result, file, indent=4)
                            return result
                        else:
                            print(f"CVE {cve_id} NVD資料中沒有映射的 CWE")
                    else:
                        print(f"CVE {cve_id} 的資料中沒有 weaknesses 訊息")
            elif response.status_code == 403:
                print(f"API 限制達到，休息一會再嘗試...")
                time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
            else:
                print(f"無法獲取 {cve_id} 的資料，狀態碼: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"獲取 {cve_id} 的資料時出錯: {e}")
            time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff

    return None



def fetch_all_cve_details(cve_ids, storage_dir, use_local_only=False):
    cve_details = []

    if use_local_only:
        # 使用本地資料夾中的所有 CVE 資料
        for cve_id in tqdm(cve_ids, desc="Fetching CVE details from local storage", disable=True):
            storage_file = os.path.join(storage_dir, f"{cve_id}.json")
            if os.path.exists(storage_file):
                with open(storage_file, 'r') as file:
                    cve_details.append(json.load(file))
        return cve_details

    # 如果不是只使用本地資料
    for cve_id in tqdm(cve_ids, desc="Fetching CVE details"):
        result = fetch_cve_details(cve_id, storage_dir)
        if result:
            cve_details.append(result)
        # 增加delay避免api被封鎖
        time.sleep(1)

    return cve_details

def filter_cve_details(cve_details):
    filtered_cve_details = [detail for detail in cve_details if extract_cwe_id(detail['CWE-ID']) in df_cwe_selected['CWE-ID'].values]
    print(f"總共有 {len(filtered_cve_details)} 個 CVE 需要比較。")
    return filtered_cve_details

def run_inference(cve_details, cve_severities):
    results = []
    severity_correct = {level: 0 for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']}
    severity_count = {level: 0 for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']}
    overall_correct = 0
    overall_count = 0
    non_2023_correct = 0
    non_2023_count = 0

    print("開始處理 CVE...")

    for detail in tqdm(cve_details, desc="處理 CVE"):
        cve_id = detail["CVE-ID"]
        cve_description = detail["description"]
        original_cwe_id = detail["CWE-ID"]
        severity = cve_severities[cve_id]
        cve_year = cve_id.split('-')[1]

        if original_cwe_id == "Unknown":
            continue

        encoded_new_cve = tokenizer_cve_cwe(cve_description, return_tensors='pt', padding=True, truncation=True, max_length=512)

        max_prob = float('-inf')
        best_cwe_id = None
        best_cwe_description = None

        for i, row in tqdm(df_cwe_selected.iterrows(), desc=f"匹配 CWE 給 {cve_id}", total=len(df_cwe_selected)):
            cwe_description = row['CWE-Description']
            combined_description = cve_description + " [SEP] " + cwe_description
            encoded_combined = tokenizer_cve_cwe(combined_description, return_tensors='pt', padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                combined_logits = model_cve_cwe(encoded_combined['input_ids'].to(device), encoded_combined['attention_mask'].to(device))
                combined_probabilities = F.softmax(combined_logits, dim=1)
                positive_prob = combined_probabilities[:, 1].item()

            if positive_prob > max_prob:
                max_prob = positive_prob
                best_cwe_id = int(row['CWE-ID'])
                best_cwe_description = cwe_description

        results.append({
            "CVE-ID": cve_id,
            "CVE-Description": cve_description,
            "Mapped CWE-ID": best_cwe_id,
            "CWE Description": best_cwe_description,
            "Positive Probability": max_prob,
            "Original CWE-ID": original_cwe_id,
            "Severity": severity
        })

        severity_count[severity] += 1
        overall_count += 1
        if cve_year != "2023":
            non_2023_count += 1
        if best_cwe_id == extract_cwe_id(original_cwe_id):
            severity_correct[severity] += 1
            overall_correct += 1
            if cve_year != "2023":
                non_2023_correct += 1

    severity_accuracy = {level: (severity_correct[level] / severity_count[level] if severity_count[level] > 0 else 0) for level in severity_count}
    overall_accuracy = overall_correct / overall_count if overall_count > 0 else 0
    non_2023_accuracy = non_2023_correct / non_2023_count if non_2023_count > 0 else 0

    print("推理結果:")
    print(results)
    print("嚴重程度準確率:")
    print(severity_accuracy)
    print("整體準確率:")
    print(overall_accuracy)
    print("非2023年準確率:")
    print(non_2023_accuracy)

    return results, severity_accuracy, overall_accuracy, non_2023_accuracy, severity_count

# 進行CVE到CAPEC的映射
def map_cve_to_capec(cve_details):
    capec_results = []

    # 獲取所有CAPEC描述和對應的CAPEC ID
    all_capec_descriptions = df_inference[['CAPEC-ID', 'CAPEC-Description']]

    for cve_detail in tqdm(cve_details, desc="處理 CVE"):
        mapped_cwe_id = cve_detail['Mapped CWE-ID']  # 這是模型推理後的CWE ID
        original_cwe_id = extract_cwe_id(cve_detail['Original CWE-ID'])  # 這是從原始數據中提取的CWE ID

        cve_id = cve_detail["CVE-ID"]
        cve_description = cve_detail["CVE-Description"]
        severity = cve_detail['Severity']

        capec_probs = []
        for _, row in tqdm(all_capec_descriptions.iterrows(), desc=f"匹配 CAPEC 給 {cve_id}", total=len(all_capec_descriptions)):
            capec_id = row['CAPEC-ID']
            capec_description = row['CAPEC-Description']

            if pd.isna(capec_description):
                continue

            combined_description = cve_description + " [SEP] " + capec_description
            encoded_combined = tokenizer_cwe_capec(combined_description, return_tensors='pt', padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                combined_logits = model_cwe_capec(encoded_combined['input_ids'].to(device), encoded_combined['attention_mask'].to(device))
                combined_probabilities = F.softmax(combined_logits, dim=1)
                positive_prob = combined_probabilities[:, 1].item()

            capec_probs.append((capec_id, capec_description, positive_prob))

        # 按概率排序並選取最高的
        top_capec_probs = sorted(capec_probs, key=lambda x: x[2], reverse=True)
        best_capec_id = top_capec_probs[0][0] if top_capec_probs else "N/A"
        best_capec_description = top_capec_probs[0][1] if top_capec_probs else "N/A"

        # 使用原始CWE ID來獲取對應的Original CAPEC ID
        original_capec_ids = df_mapping[df_mapping['CWE-ID'] == original_cwe_id]['CAPEC-ID'].tolist()
        original_capec_id = original_capec_ids[0] if original_capec_ids else "N/A"

        capec_results.append({
            "CVE-ID": cve_id,
            "CVE-Description": cve_description,
            "Severity": severity,
            "Mapped CWE-ID": mapped_cwe_id,  # 保留模型推理的CWE ID
            "CWE Description": cve_detail['CWE Description'],
            "Mapped CAPEC-ID": best_capec_id,
            "CAPEC Description": best_capec_description,
            "Original CWE-ID": cve_detail['Original CWE-ID'],  # 保留原始的CWE ID
            "Original CAPEC-ID": original_capec_id
        })

    return capec_results

def calculate_capec_accuracy(capec_results):
    total_correct = 0
    total_actual = 0

    for result in capec_results:
        original_capec_ids = result['Original CAPEC-ID'].split(', ')
        mapped_capec_id = str(result['Mapped CAPEC-ID'])

        if original_capec_ids == ["N/A"]:
            continue  # 跳過沒有 Original CAPEC-IDs 的記錄

        if mapped_capec_id in original_capec_ids:
            total_correct += 1  # 如果映射的 CAPEC ID 存在於 Original CAPEC-IDs 中，計算為正確
        total_actual += 1  # 計算總數

    accuracy = total_correct / total_actual if total_actual > 0 else 0  # 計算準確率
    print(f"CAPEC mapping accuracy: {accuracy:.4f}")
    return accuracy

def process_new_reports():
    namespace = "ricxapp"
    hardcoded_reports = [
        "replicaset-qp-h-84cdd7d847-qp-h",
        "replicaset-rc-ricxapp-deployment-764c8bffd4-ricxapp-container",
        "replicaset-ad-i-release-7cdbd655d4-ad-i-container"
    ]

    # 自動偵測存在的 vulnerabilityreport
    kubectl_command = f"kubectl get vulnerabilityreport -n {namespace} -o json"
    process = subprocess.Popen(kubectl_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    if error:
        print(f"獲取 VulnerabilityReport 時出錯: {error}")
        return

    try:
        reports = json.loads(output)
    except json.JSONDecodeError as e:
        print(f"解析 JSON 時出錯: {e}")
        return

    for report in reports['items']:
        report_name = report['metadata']['name']
        report_path = os.path.join(result_dir, f"{report_name}.json")

        if os.path.exists(report_path):
            print(f"報告 {report_name} 已經處理過，跳過。")
            continue

        print(f"處理新的報告 {report_name}...")

        # 獲取 CVE IDs
        cve_ids = [vuln['vulnerabilityID'] for vuln in report['report']['vulnerabilities'] if vuln['vulnerabilityID'].startswith('CVE')]
        cve_severities = {vuln['vulnerabilityID']: vuln['severity'] for vuln in report['report']['vulnerabilities'] if vuln['vulnerabilityID'].startswith('CVE')}

        if report_name in hardcoded_reports:
            print(f"報告 {report_name} 使用固定報告存儲資料...")
            cve_details = fetch_all_cve_details(cve_ids, storage_dir=fixed_report_storage_dir, use_local_only=True)
        else:
            print(f"報告 {report_name} 從 NVD API 獲取 CVE 資料...")
            cve_details = fetch_all_cve_details(cve_ids, storage_dir=general_storage_dir)

        filtered_cve_details = filter_cve_details(cve_details)

        cwe_results, severity_accuracy, overall_accuracy, non_2023_accuracy, severity_count = run_inference(filtered_cve_details, cve_severities)
        capec_results = map_cve_to_capec(cwe_results)

        capec_accuracy = calculate_capec_accuracy(capec_results)

        formatted_results = []
        for result in capec_results:
            formatted_results.append({
                "CVE-ID": result["CVE-ID"],
                "CVE-Description": result["CVE-Description"],
                "Severity": result["Severity"],
                "Mapped CWE-ID": result["Mapped CWE-ID"],
                "CWE Description": result["CWE Description"],
                "Mapped CAPEC-ID": result["Mapped CAPEC-ID"],
                "CAPEC Description": result["CAPEC Description"],
                "Original CWE-ID": result["Original CWE-ID"],
                "Original CAPEC-ID": result["Original CAPEC-ID"]
            })

        output_data = {
            "results": formatted_results,
            "statistics": {
                "severity_accuracy": severity_accuracy,
                "overall_accuracy": overall_accuracy,
                "non_2023_accuracy": non_2023_accuracy,
                "severity_count": severity_count,
                "capec_accuracy": capec_accuracy
            }
        }
        with open(report_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"報告 {report_name} 處理完成，結果已保存到 {report_path}。")

while True:
    process_new_reports()
    print("等待 15 秒後重新檢查新報告...")
    time.sleep(15)
