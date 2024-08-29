# 使用官方 Python images（Debian Buster Slim 版本）
FROM python:3.8-slim-buster

# 設定工作目錄
WORKDIR /app

# 複製文件到container
COPY environment.txt ./
COPY inference_data ./inference_data
COPY nvd_cve_storage_v2 ./nvd_cve_storage_v2
COPY inference_result ./inference_result
COPY vwa_result ./vwa_result
COPY auto_inference_08.py ./

# 安裝系統dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r environment.txt \
    && rm -f /app/environment.txt

# 下載並安裝 k8s
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# 設置環境變數
ENV KUBECONFIG=/home/joehu/.kube/config

# 運行程式command
CMD ["python", "auto_inference_08.py"]