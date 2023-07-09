#!/bin/bash

docker run --gpus=all --shm-size=7gb --rm -it -d --name kazuya_mitdetection -v $(pwd):/workdir -w /workdir kazuya/mitdetection:gpu /bin/bash
