FROM artstat-base:latest

VOLUME /app
VOLUME /data/shared
VOLUME /data/local

ENV PYTHONPATH=/app/src:$PYTHONPATH

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD python3 -m jupyter notebook --allow-root \
    --ip=0.0.0.0 --notebook-dir=notebooks --port=9999

EXPOSE 9999

