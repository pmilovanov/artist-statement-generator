#!/usr/bin/env bash

IMAGE=$1

if [[ -z "$IMAGE" ]]; then
echo "Usage: $0 <docker-image-name>" >&2
echo >&2
echo "where <docker-image-name> has a corresponding *.Dockerfile in docker/"
exit 1
fi

df="docker/$IMAGE/Dockerfile"
if [[ -f "$df" ]]; then
  nvidia-docker build docker/$IMAGE -t "$IMAGE"
  exit 0
fi

echo "No such dockerfile: $df" >&2
