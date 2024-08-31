import json
import os
import subprocess
import pandas as pd
import time
import re
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import re

# 檢查是否已經設置 KUBECONFIG 環境變量
if 'KUBECONFIG' not in os.environ:
    # 如果沒有設置，嘗試設置為默認的 kube config 路徑
    default_kubeconfig_path = os.path.expanduser('~/.kube/config')
    if os.path.exists(default_kubeconfig_path):
        os.environ['KUBECONFIG'] = default_kubeconfig_path
        print(f"使用默認的 KUBECONFIG 路徑: {default_kubeconfig_path}")
    else:
        print("沒有找到 KUBECONFIG，請設置 KUBECONFIG 環境變量或指定 kube config 的路徑。")
else:
    print(f"KUBECONFIG 環境變量已設置為: {os.environ['KUBECONFIG']}")

print("開始設定推理模型...")

# 加載模型和向量化器
vectorizer = joblib.load('train_model_save_cwe/vectorizer_cvecwe.joblib')
boruta_selector = joblib.load('train_model_save_cwe/boruta_selector_cvecwe.joblib')
clf = joblib.load('train_model_save_cwe/random_forest_model_cvecwe.joblib')

vectorizer_capec = joblib.load('train_model_save_capec/vectorizer_cvecapec.joblib')
boruta_selector_capec = joblib.load('train_model_save_capec/boruta_selector_cvecapec.joblib')
clf_capec = joblib.load('train_model_save_capec/random_forest_model_cvecapec.joblib')
print("模型和向量化器加載成功。")

# 加載並處理 CWE 資料集
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

# 加載capec推理和準確率計算所需的兩個不同CSV文件
file_path_inference = 'inference_data/CAPEC_Desc_CWE_254.csv'  # 用於推理的CSV
file_path_mapping = 'inference_data/CWE_Desc_CAPEC_129.csv'  # 用於準確率計算的CSV

try:
    df_inference = pd.read_csv(file_path_inference, encoding='utf-8')
    print("推理用資料集加載成功！")
except Exception as e:
    print(f"推理用資料集加載失敗: {e}")
print(df_inference.head())

try:
    df_mapping = pd.read_csv(file_path_mapping, encoding='utf-8')
    print("準確率計算用資料集加載成功！")
except Exception as e:
    print(f"準確率計算用資料集加載失敗: {e}")
print(df_mapping.head())

# 確認 vulnerabilityreport 資料夾是否存在，不存在則創建
storage_dir = 'vulnerabilityreport'
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

def extract_cwe_id(cwe_id_str):
    match = re.search(r'\d+', cwe_id_str)
    return int(match.group()) if match else None

def fetch_cve_details(cve_id):
    storage_file = os.path.join('vulnerabilityreport', f"{cve_id}.json")
    if os.path.exists(storage_file):
        with open(storage_file, 'r') as file:
            return json.load(file)
    else:
        return None

def fetch_all_cve_details(cve_ids):
    cve_details = []
    for cve_id in cve_ids:
        result = fetch_cve_details(cve_id)
        if result:
            cve_details.append(result)
    return cve_details

def filter_cve_details(cve_details):
    filtered_cve_details = [detail for detail in cve_details if extract_cwe_id(detail['CWE-ID']) in df_cwe_selected['CWE-ID'].values]
    print(f"總共有 {len(filtered_cve_details)} 個 CVE 需要比較。")
    return filtered_cve_details

# 進行cve到cwe映射
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

        encoded_new_cve = vectorizer.transform([detail["description"]])

        max_prob = float('-inf')
        best_cwe_id = None
        best_cwe_description = None

        for i, row in tqdm(df_cwe_selected.iterrows(), desc=f"匹配 CWE 給 {cve_id}", total=len(df_cwe_selected)):
            cwe_description = row['CWE-Description']
            combined_description = cve_description + " [SEP] " + cwe_description
            encoded_combined = vectorizer.transform([combined_description])

            combined_probabilities = clf.predict_proba(encoded_combined[:, boruta_selector.support_])
            positive_prob = combined_probabilities[0, 1]

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
def extract_cwe_id(cwe_id_str):
    match = re.search(r'\d+', cwe_id_str)
    return int(match.group()) if match else None

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

            # 拼接CVE描述和CAPEC描述
            combined_description = cve_description + " [SEP] " + capec_description
            encoded_combined = vectorizer_capec.transform([combined_description])

            # 使用隨機森林模型進行推理
            combined_probabilities = clf_capec.predict_proba(encoded_combined[:, boruta_selector_capec.support_])
            positive_prob = combined_probabilities[0, 1]  # 取得是映射到CAPEC的概率

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
    report_names = [
        "replicaset-qp-h-84cdd7d847-qp-h",
        "replicaset-rc-ricxapp-deployment-764c8bffd4-ricxapp-container",
        "replicaset-ad-i-release-7cdbd655d4-ad-i-container",     
    ]

    for specific_report_name in report_names:
        kubectl_command = f"kubectl get vulnerabilityreport {specific_report_name} -n {namespace} -o json"
        process = subprocess.Popen(kubectl_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(f"獲取 VulnerabilityReport 時出錯: {error}")
            continue

        report = json.loads(output)

        report_name = report['metadata']['name']
        report_path = f"{report_name}.json"

        if os.path.exists(report_path):
            print(f"報告 {report_name} 已經處理過，跳過。")
            continue

        print(f"處理新的報告 {report_name}...")

        cve_ids = [vuln['vulnerabilityID'] for vuln in report['report']['vulnerabilities'] if vuln['vulnerabilityID'].startswith('CVE')]
        cve_severities = {vuln['vulnerabilityID']: vuln['severity'] for vuln in report['report']['vulnerabilities'] if vuln['vulnerabilityID'].startswith('CVE')}

        cve_details = fetch_all_cve_details(cve_ids)
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

if not os.path.exists('vulnerabilityreport'):
    os.makedirs('vulnerabilityreport')

while True:
    process_new_reports()
    print("等待 15 秒後重新檢查新報告...")
    time.sleep(15)
