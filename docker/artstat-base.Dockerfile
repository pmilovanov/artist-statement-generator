FROM tensorflow/tensorflow:latest-gpu


RUN apt update
RUN apt install -y python3-pip

RUN pip3 install tensorflow-gpu
RUN pip3 install keras

RUN pip3 install nltk tqdm unidecode click h5py pytest
RUN pip3 install regex

RUN pip3 install terminado jupyter
RUN pip3 install jupyter-emacskeys

ENV PYTHONPATH=/app/src:$PYTHONPATH

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

