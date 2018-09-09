#!/bin/bash

DATA=$HOME/kdata

nvidia-docker run --rm -it -p 9999:9999 \
    -v $(pwd):/app \
    -v $DATA/local:/data/local \
    -v $DATA/shared:/data/shared \
    -v $HOME/gcloud-auth:/config \
    --name artstat-dev \
    artstat-dev:latest $@


    #--mount source="$HOME/kdata/shared",target=/data/shared \
