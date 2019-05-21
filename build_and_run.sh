#!/bin/bash
docker build -f docker/Dockerfile . -t sequence_classification:latest
docker run -it --rm sequence_classification:latest /bin/bash -c "python /src/sequence_classifier.py"