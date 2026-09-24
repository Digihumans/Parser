FROM python:3.12-slim

WORKDIR /app

COPY . .

ENV TZ="Asia/Kolkata"

RUN apt update && apt upgrade -y&& apt install -y --no-install-recommends build-essential ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libreoffice && \
    apt autoremove -y && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip && \
    pip install uv && \
    uv pip install --no-cache-dir -r current_requirements.txt --system && \
    docling-tools models download &&\
    pip cache purge
    # chmod +x run.sh

ENTRYPOINT ["python3","main.py"]

CMD [""]