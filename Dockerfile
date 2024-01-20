FROM python:3.10.11-slim

ENV https_proxy=http://192.168.16.101:1999 http_proxy=http://192.168.16.101:1999 no_proxy=127.0.0.0/8,192.168.0.0/16,localhost,172.0.0.0/8 share=True

RUN apt-get update && apt-get install curl wget ffmpeg ca-certificates -y \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./deploy/requirements.txt .

RUN pip3 install -r requirements.txt

RUN rm -rf /root/.cache/* /var/cache/*

EXPOSE 80

COPY ./deploy/* ./

COPY ./si-entrypoint.sh /si-entrypoint.sh

RUN chmod 700 /si-entrypoint.sh

ENV LISTEN_PORT=80

ENTRYPOINT ["/si-entrypoint.sh"]
