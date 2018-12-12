#!/bin/bash

DATA=$HOME/data

docker run --rm -it -p 9999:9999 \
    -v $(pwd):/app \
    -v $DATA/local:/data/local \
    -v $DATA/shared:/data/shared \
    -v $HOME/gcloud-auth:/config \
    --name artstat-cmle \
    artstat-cmle:latest $@


    #--mount source="$HOME/kdata/shared",target=/data/shared \
