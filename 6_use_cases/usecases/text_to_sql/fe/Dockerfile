#FROM --platform=linux/x86-64 python:3.11.1-slim
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app.py app.py
COPY const.py const.py
COPY utils.py utils.py
COPY schema.png schema.png
COPY populate.sql populate.sql

EXPOSE 80

ENTRYPOINT [ "streamlit", "run", "app.py", \
    "--server.enableCORS", "true", \
             "--server.port", "80", \
             "--browser.serverPort", "80"]