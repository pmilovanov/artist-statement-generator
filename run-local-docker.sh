#!/bin/bash

DATA=$HOME/data

nvidia-docker run --rm -it -p 9999:9999 \
    -v $(pwd):/app \
    -v $DATA/local:/data/local \
    -v $DATA/shared:/data/shared \
    --name artstat-dev \
    artstat:latest $@


    #--mount source="$HOME/kdata/shared",target=/data/shared \
