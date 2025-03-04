FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /usr/src/app

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get upgrade -y

COPY *requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY *.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]