FROM artstat-base:latest

# python3
VOLUME /app
VOLUME /data/shared
VOLUME /data/local
VOLUME /config

ENV PYTHONPATH=/app/src:$PYTHONPATH

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD python -m jupyter notebook --allow-root \
    --ip=0.0.0.0 --notebook-dir=notebooks --port=9999

EXPOSE 9999


#####################

RUN pip install google-cloud-datastore

