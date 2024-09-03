import os
import pdfplumber
import re
from docx import Document
import warnings
import logging
import spacy
from collections import Counter
from nltk import word_tokenize, bigrams
import time

import nltk
nltk.download('punkt')

# 隱藏無關警告
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")

# 加載英文模型
nlp = spacy.load("en_core_web_sm")

# 記錄被跳過的文件名
skipped_files = []

# 處理PDF文件
def process_pdf(file_path, min_font_size=8, max_font_size=12):
    print(f"正在處理PDF文件: {file_path}")
    all_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = ""
                for char in page.chars:
                    if min_font_size <= char['size'] <= max_font_size:
                        page_text += char['text']
                all_text += page_text + "\n"
    except Exception as e:
        print(f"處理PDF文件時出錯: {file_path}, 錯誤: {e}")
        skipped_files.append(file_path)
    return all_text

# 處理Word文件
def process_word(file_path, min_font_size=8, max_font_size=12):
    print(f"正在處理Word文件: {file_path}")
    all_text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.runs and all(min_font_size <= run.font.size.pt <= max_font_size for run in para.runs if run.font.size):
                all_text += para.text + "\n"
    except Exception as e:
        print(f"處理Word文件時出錯: {file_path}, 錯誤: {e}")
        skipped_files.append(file_path)
    return all_text

# 檢測檔案中是否有緊密連在一起的字
def has_tightly_packed_text(text):
    return bool(re.search(r'\w{1000,}', text))

# 要讀取的資料夾
directories = ['3GPP_SA3']

# 移除不重要內容的函數
def remove_irrelevant_content(text):
    text = extract_main_content(text)
    text = remove_table_of_contents(text)
    text = remove_headers_and_footers(text)
    text = remove_lists_tables_headers(text)
    text = clean_text(text)
    return text

# 移除目錄
def remove_table_of_contents(text):
    lines = text.split('\n')
    if not any('........' in line for line in lines):
        print("檢查目錄標誌未找到，跳過處理")
        return text

    content_removed = []
    toc_end = False
    for line in lines:
        if '........' in line:
            toc_end = True
            continue
        if not toc_end:
            continue
        content_removed.append(line)
    print(f"Processed text length: {len(content_removed)}")
    return '\n'.join(content_removed)

# 移除頁眉和頁腳
def remove_headers_and_footers(text):
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'ETSI TS \d+ \w+\n?', '', text)
    text = re.sub(r'\n?ETSI', '', text)
    text = re.sub(r'\n?\d{4}', '', text)
    text = re.sub(r'Your use is subject to the copyright statement.*\n', '', text, flags=re.IGNORECASE)
    return text

# 移除列表、表格和標題
def remove_lists_tables_headers(text):
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(^|\n)([-•] )', '\n', text)
    text = re.sub(r'\n[A-Z ]+\n', '\n', text)
    text = re.sub(r'\n\d+\.\d+.*\n', '\n', text)
    text = re.sub(r'\n_*[^\n]+\n_*', '\n', text) # 移除帶下劃線的部分
    return text

# 移除文件不重要的細項
def clean_text(text):
    # 定義要去除的模式
    patterns = [
        r'^\W*\d+\W*',  # 移除行首的非字母字符和數字（行號或章節號）
        r'Tel\.:.*?Fax:.*?\n',  # 移除包含電話號碼和傳真號碼的行
        r'Siret N°.*?\n',  # 移除公司的識別號碼
        r'NAF.*?\n',  # 移除行業分類代碼
        r'https?://\S+',  # 移除URL網頁
        r'\b3GPP TS \d+\.\d+ version \d+\.\d+\.\d+',  # 移除3GPP版本號
        r'\d{4}GPP TS \d+\.\d+',  # 移除4位數開頭的版本號
        r'GPP TS \d+\.\d+',  # 移除GPP TS開頭的版本號
        r'Copyright.*?O-RAN ALLIANCE e\.V\..*?\n',  # 移除版權聲明
        r'All rights reserved.*?\n',  # 移除所有權保留聲明
        r'This document.*?\n',  # 移除文檔說明
        r'Your use is subject to copyright statement.*?\n',  # 除版權使用聲明
        r'For a specific reference.*?\n',  # 移除特定引用提示
        r'\[\d+\]',  # 移除文獻中的引用標記
        r'O-RAN.WG[\d.]+.*?\n',  # 移除O-RAN.WG開頭的部分
        r'__+.*?__+',  # 移除長下劃線部分
        r'(© by the O-RAN ALLIANCE e\.V\. )+',  # 移除版權聲明
        r'This Technical Specification.*?\n',  # 移除技術規範說明
        r'Version \d+\.\d+\.\d+.*?\n',  # 移除版本號
        r'v\d+\.\d+\.\d+',  # 移除簡短版本號
        r'Release \d+.*?\n',  # 移除Release信息
        r'V\d+\.\d+\.\d+',  # 移除大寫版本號
        r'\bversion \d+\.\d+\.\d+',  # 去除獨立的版本號
        r'3GPP TS \d+\.\d+ version \d+\.\d+\.\d+',  # 去除3GPP的版本號
        r'Release \d+ of the present document has been produced by the 3rd Generation Partnership Project 3GPP.',  # 去除特定的發布說明
        r'z = the third digit is incremented when editorial only changes have been incorporated in the document.',  # 去除版本更新說明
        r'\d{4}GPP TS \d+\.\d+ Release \d+This Technical Specification has been produced by the 3rd Generation Partnership Project 3GPP.',  # 移除特定格式的技術說明
        r'ying change of release date and an increase in version number as follows Version x.y.z where x the first digit 1 presented to TSG for information 2 presented to TSG for approval 3 or greater indicates TSG approved document under change control',  # 去除特定說明
        r'Release \d+',  # 移除發布號
        r'Version \d+\.\d+\.\d+',  # 移除版本號
        r'\b(?:shall|should|may|must)\b.*?[.!?](?:\s|$)',  # 移除規範性聲明
        r'3GPP.*?\.',  # 移除3GPP引用
        r'\bTS \d+\.\d+\b',  # 移除TS標號
        r'\bETSI\b',  # 移除ETSI
        r'Document history.*?(?=^\w{2,})',  # 移除文檔歷史部分
        r'\bRFC \d+\b',  # 移除RFC引用
        r'Security architecture.*?(?=^\w{2,})',  # 移除安全架構描述
        r'Figure \d+\.\d+.*?(?=^\w{2,})',  # 移除圖表描述
        r'\d+\.\d+\.\d+.*?(?=^\w{2,})',  # 移除編號後的描述
        r'\b(?:Annex|Appendix)\s+\w+.*?(?=^\w{2,})',  # 移除附錄
        r'Copyright.*?(?=^\w{2,})',  # 移除版權聲明
        r'\s{2,}',  # 替換多個空格為一個
        r'\n+',  # 替換多個換行符為一個
        r'^\s*[A-Za-z0-9]\s*$',  # 移除只剩單個字符的行
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 清理空行和多餘的空格
    text = re.sub(r'\s{2,}', ' ', text)  # 將連續的多個空格替換為單個空格
    text = re.sub(r'\n+', '\n', text)    # 連續的多個換行符替換為單個換行符
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 提取內文部分
def extract_main_content(text):
    start_pattern = r'3 Definitions and abbreviations'
    end_pattern = r'Annex A \(informative\)|Change history|History'
    # 提取內文
    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)
    if start_match and end_match:
        return text[start_match.start():end_match.start()]
    elif start_match:
        return text[start_match.start():]
    return text

# 提取完整句子
def extract_complete_sentences(text):
    doc = nlp(text)
    sentences = []
    
    for sent in doc.sents:
        sentence = sent.text.strip()
        if (
            10 < len(sentence.split()) <= 25  # 句子長度限制在10到25個單詞之間
            and not re.match(r'^[\w\s,]*$', sentence)  # 過濾掉只有單詞或簡單短語的句子
            and not sentence.lstrip().startswith(tuple('0123456789'))  # 過濾掉以數字開頭的句子
            and re.search(r'[.!?]$', sentence)  # 確保句子結尾有標點符號
        ):
            # 檢查語法完整性
            root = [token for token in sent if token.head == token]  # 找到根節點
            if root and root[0].pos_ in ['VERB', 'NOUN']:  # 根節點應為動詞或名詞
                sentences.append(sentence)

    print("前6句完整句子:")
    for i, sentence in enumerate(sentences[:6]):
        print(f"提取到完整句子: {sentence}")
    
    print("後6句完整句子:")
    for i, sentence in enumerate(sentences[-6:]):
        print(f"提取到完整句子: {sentence}")
    
    return "\n".join(sentences)

# 處理文件並保存清洗後的文本
def clean_and_save(file_path, min_size=5000):
    file_text = process_file(file_path)
    if not file_text or has_tightly_packed_text(file_text):
        print(f"跳過字連在一起的檔案: {file_path}")
        skipped_files.append(file_path)
        return

    cleaned_text = remove_irrelevant_content(file_text)
    cleaned_text = extract_complete_sentences(cleaned_text)
    if len(cleaned_text) < min_size:
        print(f"跳過字數不足的檔案: {file_path}")
        skipped_files.append(file_path)
        return

    save_path = os.path.join(processed_data_dir, os.path.basename(file_path) + ".txt")
    with open(save_path, 'w') as f:
        f.write(cleaned_text)
    print(f"已保存清洗後的文本到: {save_path}")

# 處理PDF和Word文件的函數
def process_file(file_path, min_font_size=8, max_font_size=12):
    if file_path.endswith('.pdf'):
        return process_pdf(file_path, min_font_size, max_font_size)
    elif file_path.endswith('.docx'):
        return process_word(file_path, min_font_size, max_font_size)
    return None

# 提取專業術語
def extract_special_terms(texts, common_word_limit=1000, common_bigram_limit=500):
    word_counts = Counter()
    bigram_counts = Counter()

    for text in texts:
        tokens = word_tokenize(text)
        word_counts.update(tokens)
        bigram_counts.update(bigrams(tokens))

    common_words = word_counts.most_common(common_word_limit)
    common_bigrams = bigram_counts.most_common(common_bigram_limit)

    special_terms = set()
    for word, count in common_words:
        if re.search(r'\d', word) or len(word) > 15: # 排除包含數字或過長的詞語
            continue
        special_terms.add(word)

    for bigram, count in common_bigrams:
        term = ' '.join(bigram)
        if re.search(r'\d', term) or len(term) > 30: # 排除包含數字或過長的bigram
            continue
        special_terms.add(term)

    return list(special_terms)

# 創建存儲清洗數據的資料夾
processed_data_dir = "3GPP_SA3_cleantext"
os.makedirs(processed_data_dir, exist_ok=True)

# 紀錄開始時間
start_time = time.time()

# 清洗所有文件
all_texts = []
for directory in directories:
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        clean_and_save(file_path)
        cleaned_file_path = os.path.join(processed_data_dir, os.path.basename(file_path) + ".txt")
        if os.path.exists(cleaned_file_path):
            with open(cleaned_file_path, 'r') as f:
                all_texts.append(f.read())
        else:
            print(f"文件不存在，跳過: {cleaned_file_path}")

# 提取專業術語
special_terms = extract_special_terms(all_texts)
print(f"提取到的專業術語: {special_terms}")

end_time = time.time()
print(f"所有文件清洗完成。總耗時: {end_time - start_time} 秒")
print(f"被跳過的文件數量: {len(skipped_files)}")
print(f"被跳過的文件檔案名稱: {skipped_files}")
