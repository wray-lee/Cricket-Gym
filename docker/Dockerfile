# 使用轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (如有需要，例如 gcc)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码 (在 Docker Compose 中会被 Volume 覆盖，但对生产部署很重要)
COPY . .

# 设置 PYTHONPATH
ENV PYTHONPATH=/app

# 默认命令
CMD ["python", "main.py"]