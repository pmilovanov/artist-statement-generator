FROM tensorflow/tensorflow:1.10.0-py3
#FROM tensorflow/tensorflow:latest-gpu-py3

ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install python3.6
#RUN pip install keras
RUN pip install nltk tqdm unidecode click h5py pytest
RUN pip install regex

RUN pip install terminado jupyter
RUN pip install jupyter-emacskeys

RUN pip install google-cloud-datastore google-cloud-storage
RUN pip install python-box

################################################################################
WORKDIR /app

VOLUME /app
VOLUME /data
VOLUME /config

ENV PYTHONPATH=/app/src:$PYTHONPATH

ENV GOOGLE_APPLICATION_CREDENTIALS="/config/pm-artstat-auth.json"

CMD python -m jupyter notebook --allow-root \
    --ip=0.0.0.0 --notebook-dir=notebooks --port=9999

EXPOSE 9999








