#!/bin/bash
docker build \
    --pull \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USER=hoge --build-arg PASSWORD=fuga \
    -t kazuya/mitdetection:gpu ./docker/gpu