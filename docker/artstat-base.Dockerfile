FROM tensorflow/tensorflow:latest-gpu-py3


# Install python3.6
RUN pip install keras
RUN pip install nltk tqdm unidecode click h5py pytest
RUN pip install regex

RUN pip install terminado jupyter
RUN pip install jupyter-emacskeys

ENV PYTHONPATH=/app/src:$PYTHONPATH

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

