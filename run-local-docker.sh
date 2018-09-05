#!/bin/bash

DATA=$HOME/kdata

nvidia-docker run -d -it -p 9999:9999 \
    --rm \
    -v $(pwd):/app \
    -v $DATA/local:/data/local \
    -v $DATA/shared:/data/shared \
    --name artstat-dev \
    artstat:latest $@


    #--mount source="$HOME/kdata/shared",target=/data/shared \
