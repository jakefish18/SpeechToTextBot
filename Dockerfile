FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Системные зависимости: ffmpeg для конвертации аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# torch уже есть в базовом образе, убираем из requirements чтобы не переустанавливать
RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app

# Сначала зависимости (кешируются отдельно)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY . .

# Папка для временных файлов
RUN mkdir -p temp

CMD ["python", "run.py"]
