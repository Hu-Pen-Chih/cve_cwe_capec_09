import os
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 定義各個工作組的URL和下載目錄
urls = {
    "SA1": 'https://www.3gpp.org/ftp/Specs/archive/22_series/',
    "SA2": 'https://www.3gpp.org/ftp/Specs/archive/23_series/',
    "SA3": 'https://www.3gpp.org/ftp/Specs/archive/33_series/',
    "SA4": 'https://www.3gpp.org/ftp/Specs/archive/26_series/',
    "SA5": 'https://www.3gpp.org/ftp/Specs/archive/32_series/'
}

def create_download_dir(group):
    download_dir = f'3GPP_{group}'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    return download_dir

def fetch_files(base_url):
    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [link.get('href') for link in soup.find_all('a')]
        return links
    except Exception as e:
        print(f"An error occurred while fetching files from {base_url}: {e}")
        return []

def extract_file_names(links, base_url):
    file_names = []
    for link in links:
        if link.startswith(base_url) and not link.endswith('/'):
            file_name = link.replace(base_url, '')
            file_names.append(file_name)
    return file_names

def download_file(file_url, download_path):
    try:
        response = requests.get(file_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f'Downloaded {file_url} to {download_path}')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading {file_url}: {e}')

def extract_zip(file_path, extract_to):
    try:
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f'Extracted {file_path} to {extract_to}')
    except Exception as e:
        print(f'Error extracting {file_path}: {e}')

def process_folder(base_url, folder_name, download_dir):
    folder_url = base_url + folder_name + '/'
    
    try:
        print(f'Trying to access folder {folder_url}')
        folder_response = requests.get(folder_url, headers=headers)
        folder_response.raise_for_status()
        print(f'Successfully accessed folder {folder_url}')
    except requests.exceptions.RequestException as e:
        print(f'Error accessing folder {folder_url}: {e}')
        return

    folder_soup = BeautifulSoup(folder_response.text, 'html.parser')
    files = folder_soup.find_all('a', href=lambda x: x and x.endswith('.zip'))
    if files:
        print(f'Found {len(files)} files in folder {folder_url}')
        for file in files:
            print(f'File: {file.get("href")}')

        sorted_files = sorted(files, key=lambda x: x.get('href').split('/')[-1], reverse=True)
        latest_file = sorted_files[0] if sorted_files else None
        
        if latest_file:
            file_href = latest_file.get('href').split('/')[-1]
            file_url = folder_url + file_href
            file_name = file_href
            download_path = os.path.join(download_dir, file_name)
            
            print(f'Latest file to download: {file_url}')
            
            download_file(file_url, download_path)
            extract_zip(download_path, download_dir)
            os.remove(download_path)
            print(f'Removed the zip file: {download_path}')

print('所有文件下載並解壓縮完成。')

for group, base_url in urls.items():
    print(f"\nFetching files for {group} from {base_url}")
    links = fetch_files(base_url)
    if links:
        folder_names = extract_file_names(links, base_url)
        print(f"Found folder names for {group}: {folder_names}")

        download_dir = create_download_dir(group)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_folder, base_url, folder_name, download_dir) for folder_name in folder_names]
            for future in as_completed(futures):
                future.result()
    else:
        print(f"No files found for {group}.")

print('所有文件下載並解壓縮完成。')