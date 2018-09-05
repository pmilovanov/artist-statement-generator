FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
CMD python3 -m jupyter notebook --allow-root \
    --ip=0.0.0.0 --notebook-dir=notebooks --port=9999

RUN apt update

RUN apt install -y python3-pip

VOLUME /app
VOLUME /data/shared
VOLUME /data/local

#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3 get-pip.py

RUN pip3 install nltk tqdm unidecode click h5py aiofiles pygtrie pytest


RUN pip3 install regex

RUN pip3 install terminado jupyter
RUN pip3 install jupyter-emacskeys

RUN pip3 install tensorflow-gpu

ENV PYTHONPATH=/app/src:$PYTHONPATH

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 9999

