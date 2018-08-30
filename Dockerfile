FROM tensorflow/tensorflow:latest-gpu


RUN apt update

RUN apt install -y python3-pip

VOLUME /app
VOLUME /data/shared
VOLUME /data/local

#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3 get-pip.py

RUN pip3 install nltk tqdm unidecode click

RUN pip3 install jupyter-emacskeys
RUN pip3 install regexp

ENV PYTHONPATH=/app/src:$PYTHONPATH

